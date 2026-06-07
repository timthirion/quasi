---
name: close-plan
description: Close an implementation plan (plans/NNNN-*.md) by orchestrating the review agents, refusing to close on unaddressed P0 attacks, ticking all milestones, and committing+pushing the closure. Composes pre-flight + commit-and-push + plan-skeptic + code-attacker/defender + render-attacker/defender. Research plans (RNNNN-*.md) have a separate closure path.
version: 0.1.0
---

# Close plan

The orchestrator for plan closure. Where `commit-and-push`
handles the *push* half of any unit of work, `close-plan`
handles the *was-this-actually-finished?* half — the review
gauntlet that the four agent roles exist to run.

A plan that flips to `completed` without this skill's gauntlet
risks shipping with a hand-waved milestone, a P0 attack
nobody surfaced, or a hero-gallery render that the agent
normalised away as "looks fine." The skill exists because
that risk has bitten this repo before.

## Inputs

* A plan path or number: `0019` or
  `plans/0019-mikktspace.md` — either is fine.
* The current branch should be the one carrying the
  plan-closing commits (typically `main`, sometimes a feature
  branch).

## The sequence (defaults)

1. **Read the plan.** Confirm `Status:` is currently `draft`
   or `active`. If already `completed`, refuse and surface
   the existing status. If `abandoned`, refuse (abandoned
   plans don't get closed; they get archived).

2. **Identify the plan-closing commit(s).** Use
   `git log --oneline --first-parent -- plans/<plan>.md` to
   find the commits that touched the plan file; the most
   recent one is usually the plan-closing edit. Diff that
   commit (or commit range) for the next steps.

3. **Invoke [`plan-skeptic`](../agents/plan-skeptic.md)** on
   the plan file. The agent reads the plan and writes its
   attack. Surface the full report.

4. **Inspect the diff for code changes.** If the
   plan-closing commit(s) touch `.rs`, `.wgsl`, `.toml`, or
   `.py` files (i.e., real code, not just plan markdown
   + new reference renders + new scene assets), invoke
   the [`code-attacker`](../agents/code-attacker.md) +
   [`code-defender`](../agents/code-defender.md) pair on
   the diff range. Surface both reports.

5. **Inspect the diff for hero-gallery PNG changes.** Hero
   gallery is the set of images embedded in the top-of-README
   `<table>` — currently:
   * `data/output/outdoor_normal_bunny_reference.png`
   * `data/output/cornell_glass_bunny_reference.png`
   * `data/output/cornell_foggy_room_reference.png`
   * `data/output/cornell_cloud_reference.png`
   * `data/output/denoise_comparison.png`

   If any of these PNGs changed in the plan-closing commit,
   materialise the *prior* committed version
   (`git show HEAD~1:<path> > /tmp/old.png`) and invoke
   [`render-attacker`](../agents/render-attacker.md) +
   [`render-defender`](../agents/render-defender.md) on each
   (old, new) pair. Surface all reports.

6. **Surface the synthesised verdict.** Read all the agent
   reports; identify every P0 attack from `plan-skeptic`,
   `code-attacker`, and `render-attacker`; identify every
   P0 attack the `code-defender` or `render-defender`
   accepted. Present to the human (or calling agent) as:

   ```
   close-plan synthesis on <plan>:
     plan-skeptic:    <N>×P0, <N>×P1
     code-attacker:   <N>×P0, <N>×P1
     code-defender:   accepted <N> P0, refuted <N>, deferred <N>
     render-attacker: <N>×P0, <N>×P1 (per scene pair)
     render-defender: <N> headline improvements
   ```

7. **Refuse to close on unaddressed P0.** If any P0 attack
   survived the defender's response (i.e., the defender
   didn't refute or properly defer it), the skill refuses
   to flip status to `completed`. The caller addresses the
   attack (fixes the code, re-renders, edits the plan) and
   re-invokes.

8. **On approval — explicit only.** Once the human (or the
   calling agent with appropriate authority — see
   [[feedback_autonomy]]) approves the closure, **only
   then**:
   * Tick all `[ ]` to `[x]` in the plan's milestone
     sections.
   * Change `Status:` to `completed`.
   * Update `Last updated:` to today's ISO date.
   * Update `Last touched on:` to a short description of
     the closing pass.

9. **Invoke [`commit-and-push`](commit-and-push.md)** with
   the plan-closure edit. The commit message references
   the closed plan + the agent reports' headline findings.
   `commit-and-push` runs pre-flight, commits, pushes,
   waits for CI.

10. **Update auto-memory.** Append a one-line entry to
    `~/.claude/projects/-Users-tt-src-quasi/memory/project_active_plans.md`
    naming the closed plan + its headline outcome. Memory
    drift on closed plans is the most common
    auto-memory-staleness failure mode this repo has had.

## Skipping or extending the defaults

Plans can declare exceptions in their `Done when` section:

* **"close-plan must invoke `ultrareview`"** — for plans that
  touch WGSL integrator math, run an `ultrareview`-style
  agent (when that skill exists) in addition to the default
  pair. Surface the report alongside the others.
* **"close-plan may skip render-attacker/defender"** — for
  plans that don't change visual output (CI workflow plans,
  README rewrites). The skill still inspects the diff for
  PNG changes but accepts a `none-detected` outcome
  silently.

Default behaviour applies when the plan declares no
exceptions.

## Refusal conditions

Hard refusals (the skill does **not** ask permission to
override):

* Plan status is already `completed` or `abandoned`.
* The plan-closing diff doesn't exist (no commit has
  touched the plan file recently — the plan isn't actually
  closeable yet).
* Any P0 attack from `plan-skeptic`, `code-attacker`, or
  `render-attacker` is unaddressed (i.e., the corresponding
  defender accepted it but proposed no fix in this pass,
  OR didn't address it at all).

Soft refusals (the skill flags but the caller can override
with explicit confirmation):

* The `Done when` section's criteria look only partially
  satisfied — the agent's judgement isn't substitute for the
  human's, but the agent surfaces the doubt.
* The plan was never `active` (jumping straight from
  `draft` to `completed` skips an intended workflow stage).

## Output shape

On a clean close:

```
close-plan <plan> on <branch>:
  ✓ plan-skeptic: 0 P0, 2 P1 (both pre-acknowledged in plan)
  ✓ code-attacker/defender: 1 P0 raised, defender refuted (cited
    pathtrace.rs:411), 3 P1 deferred to plan 00NN
  ✓ render-attacker/defender (3 scene pairs):
    - cornell_glass_bunny: no regressions; 1 improvement
    - cornell_foggy_room: no changes detected
    - outdoor_normal_bunny: 1 P1 regression (defender refuted)
  ✓ ticked 17 milestones; status → completed
  ✓ commit-and-push: CI green (run <id>, <duration>)
  ✓ memory: appended one-line entry to project_active_plans.md
Done.
```

On refusal (unaddressed P0):

```
close-plan <plan>: REFUSED
  ✗ code-attacker raised P0 at src/pathtrace.rs:411 ("emissive
    pick prob can be NaN when both env and triangle totals are
    zero"); code-defender accepted but proposed no fix in this
    pass.
Action: address the P0 (fix the code OR add a documented
guard), re-pre-flight, re-invoke close-plan.
```

## Implementation vs research plans

This skill targets **implementation plans** (`plans/NNNN-*.md`).
Research plans (`plans/research/RNNNN-*.md`) have different
closure semantics:

* Status moves to `accepted` (paper accepted) or `abandoned`
  (didn't pan out) — not `completed`.
* The `Findings` section must carry at least one entry before
  closure. Skill refuses without one.
* Defaults: invoke `research-critic` (not `plan-skeptic`).
  No code/render reviews — research plans rarely close with
  matching code commits.

A future `close-research-plan.md` skill (not scaffolded today)
would mirror this skill's shape for the research-plan case.
For now, research-plan closures are hand-driven.

## What this skill is NOT

* Not a code review for routine commits. Plain commits go
  through `commit-and-push`. `close-plan` is the
  review-gauntlet specifically for closing a plan.
* Not an unconditional approve-and-merge. The refusal
  conditions are load-bearing — if every plan-closing pass
  rubber-stamps, the skill is doing no work and should be
  removed.
* Not a substitute for human judgement. The agents produce
  reports; the human (or the appropriately authorised
  calling agent) renders the verdict. The skill enforces
  process, not taste.

## When to invoke

* When a plan's milestones are all implemented and the
  closing commit (the one that flips the plan's milestone
  ticks + status) is the next intended action. Specifically:
  before the closing edit, not after.
* When a plan has been `active` for ≥2 weeks and the
  caller wants to audit "is this actually finished, and
  what would block calling it done?" The skill produces a
  diagnostic report even when it refuses to close.

Don't invoke on draft plans that haven't been implemented
yet — the agents won't have a diff to attack and the report
is empty.
