# plans/

Living planning documents for Quasi (Rust), versioned in the repo so work can
move between machines without losing state.

## Why these live in the repo

A plan is the hand-off: anyone — human or agent — picking the repo up on another
machine should be able to read the active plan and continue without reconstructing
context. Keep plans current as you work; a ticked checkbox here is the source of
truth for "what's done." Commit plan updates alongside the code they describe.

## Layout

- `ROADMAP.md` — the high-level, phased direction. The north star.
- `NNNN-short-slug.md` — one document per concrete piece of work, zero-padded and
  incrementing (`0001-`, `0002-`, …). The number is ordering, not priority.

## Plan document template

```markdown
# <Title>

- **Status:** proposed | active | blocked | done | abandoned
- **Last updated:** YYYY-MM-DD
- **Last touched on:** <machine / context, so the next session knows where it ran>

## Goal
One paragraph: what this delivers and why it matters for the roadmap.

## Context
What exists today, relevant files, constraints, prior decisions.

## Design
The approach. Struct sketches, WGSL/pipeline shapes, trade-offs considered.

## Steps
- [ ] Concrete, checkable tasks in order. Tick as you go.

## Open questions
Unresolved decisions. Resolve and record the answer rather than deleting.

## Done when
The acceptance criteria — tests, reference images, perf targets.
```

## Conventions

- Update **Status** and **Last updated** every working session.
- Resolve an open question in-doc (with the answer) rather than dropping it.
- When a plan is `done`, leave it as a record and link it from `ROADMAP.md`.
- Render-quality work should cite a reference (ground-truth image, paper, metric)
  so correctness is verifiable on any machine.
- Native and web builds are both first-class: a plan isn't done until it works in
  both targets (unless explicitly native-only, e.g. the verification harness).
