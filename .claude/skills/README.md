# `.claude/skills/`

Project-scoped skills for agents working on Quasi.

A **skill** is a procedural recipe with embedded judgement —
the right defaults, the right failure-handling, the right
output shape — packaged so an agent can invoke it by name
instead of re-deriving the procedure each time. Skills aren't
just CLI aliases; if all the file would say is "run command
X," it doesn't earn its place here. The bar is: skills bundle
*knowledge*, not just incantations.

## Layout

Skills are **always** directory-based, even when they have
no supporting assets:

```
.claude/skills/
├── README.md              ← this file (convention)
└── <skill-name>/
    └── SKILL.md           ← the procedural recipe (required)
```

The flat-file form (`.claude/skills/<name>.md`) does **not
register** with the Claude Code harness — we tried it and the
skills were invisible to the `Skill` tool until restructured
into `<name>/SKILL.md`. Agents (under `.claude/agents/`) use
flat files; skills don't. Asymmetric convention; both are real.

A skill that grows supporting assets (templates, scripts,
prompt fragments) co-locates them in the same directory:

```
.claude/skills/<skill-name>/
├── SKILL.md               ← the procedural recipe
├── references/            ← supplementary docs the skill cites
└── templates/             ← reusable fragments the recipe references
```

Start with just `SKILL.md`. Add the auxiliary directories only
when the recipe genuinely needs to reach for one.

## Frontmatter

Every `SKILL.md` starts with YAML frontmatter:

```yaml
---
name: skill-name
description: one-line summary used at selection time
version: 0.1.0
---
```

`name` is the identifier the agent invokes (`Skill(skill:
"skill-name")`). `description` is what the agent reads to
decide whether the skill applies — make it concrete enough
that the matching call is unambiguous. ("Run pre-flight quality
gates" beats "Check the project.") `version` follows semver;
bump when the procedural recipe changes shape (not on every
typo fix).

## When a skill earns its file

* The workflow is invoked **repeatedly**, in roughly the same
  shape every time.
* There's a **right way** to do it (not just one of many).
* The procedure embeds **judgement** — defaults, failure
  handling, retries — that a fresh agent would otherwise have
  to rediscover.
* The recipe **doesn't change often**. Skills go stale fast if
  they pin to fast-moving CLI surfaces.
* The whole thing fits on **one page of markdown**. Longer than
  that and it wants to be a plan or a script, not a skill.

If a candidate skill fails any of those, prefer one of:

* A plain shell command (no encoding needed).
* A script in `scripts/` (deterministic, no agent in the
  loop).
* An implementation plan in `plans/` (multi-step,
  feature-shaped).
* A research plan in `plans/research/` (hypothesis-shaped).

## Cross-references

Skills that enforce policy from auto-memory should cite the
relevant memory entry by `[[name]]`. Skills that codify a plan's
"Done when" criteria should link the plan. The goal is the
skill is **traceable to its rationale**, not a free-floating
recipe.

## Currently scaffolded

- [`pre-flight/`](pre-flight/SKILL.md) — full quality-gate
  sequence required before any commit + push. Codifies
  [[feedback_verify_ci_after_push]].
- [`commit-and-push/`](commit-and-push/SKILL.md) — pre-flight,
  then commit with project-convention message, push, watch
  CI on the just-pushed run, refuse to consider the push
  "done" until CI is green. Codifies the green-CI half of
  [[feedback_autonomy]].
- [`close-plan/`](close-plan/SKILL.md) — orchestrator for
  closing an implementation plan. Invokes `plan-skeptic`,
  conditionally invokes `code-attacker/defender` (on code
  diffs) and `render-attacker/defender` (on hero-gallery
  PNG diffs), refuses to close on unaddressed P0 attacks,
  ticks milestones, calls `commit-and-push`. Research-plan
  closure is a separate path.
