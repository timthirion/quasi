# `.claude/agents/`

Project-scoped agent role definitions for Quasi. Where
`.claude/skills/` is **how** to do something procedurally,
`.claude/agents/` is **who** does it — a mandate, an input
contract, an output shape, and a list of anti-patterns the
role must refuse.

Agents here are **review-shaped, not implementer-shaped**.
They read code, plans, research, or renders and produce
written reports. They don't commit, push, or modify files.
The implementer agent (or human) decides what to do with the
report.

## When an agent role earns its file

* The work has a **failure mode a single fresh agent
  consistently misses** — bias toward the artifact under
  review (the drafter doesn't see their own hand-waving), or
  toward subjective normalization (a single visual judgement
  has high variance + high confidence, a dangerous mix).
* The mandate is **asymmetric**: the agent has to refuse to
  return "looks good" even when the artifact is strong. The
  point is to find the strongest available attack / defence,
  not to render a balanced verdict.
* The role can be **invoked repeatedly** across many
  artifacts — research-critic across many research plans,
  plan-skeptic across many implementation plans. Roles that
  fit only one artifact aren't roles, they're prompts.

## Adversarial / paired roles

Agents that come in pairs (code-attacker + code-defender,
render-attacker + render-defender) have opposite mandates.
**The asymmetry is the entire point.** A single agent asked
to "review this code and find issues" will produce a
balanced critique; the attacker / defender pair produces a
*sharper* one because each agent's mandate prevents the
hedging the single-agent version drifts toward.

The caller is responsible for **synthesising the verdict** —
read the attacker's report, read the defender's response,
decide what to act on. No agent in the pair is allowed to
produce the verdict; their job is to argue, not to judge.

## Frontmatter

```yaml
---
name: agent-role-name
description: One-sentence mandate. Concrete enough that the matching call is unambiguous.
tools: Read, Bash, Grep, Glob
---
```

`tools` is an **explicit comma-separated list** of tool names
(or equivalently a YAML array `["Read", "Bash", ...]`). The
prose form `tools: All tools except X` is **display-only** —
it appears in the system prompt's rendering of registered
agents, but the harness does NOT accept it in the `.md`
frontmatter. We learned this the hard way: the first version
of these files used `tools: All tools except Edit, Write,
NotebookEdit, Agent` and the registered agents came back with
the *inverse* (Edit / Write / NotebookEdit only and no read
tools). Fixed at commit (search the log for "agent tools
config" — the commit that lands alongside this convention
note).

The default tool list for review-shaped agents in this
directory is `Read, Bash, Grep, Glob` — enough to read code,
plans, renders, and run `git diff` / `git log` / `git show`
without modifying anything. `research-critic` additionally
gets `WebSearch, WebFetch` for prior-art lookup. Add or
remove per role; the choice should be deliberate.

**Never include `Edit, Write, NotebookEdit, Agent` in a
review-shaped agent's tools list.** Edit / Write would let
the agent modify the artifact it's reviewing (a recurrent
self-deception failure mode: the implementer-mind "fixes"
the attack mid-flight rather than scoring it). `Agent` would
let it recurse, blowing the budget and tangling the
synthesis path.

## Required body sections

```
## Mandate
The one-sentence claim the agent must defend. Asymmetric for
adversarial roles.

## Inputs
What artifact the caller passes in. Be specific — a file
path, a commit range, a pair of PNGs. Vague inputs produce
vague reports.

## Output shape
The structured form the report takes. Bullet lists, severity
tags, specific deliverables. Agents under-produce when the
output shape isn't pinned.

## Anti-patterns
The failure modes the agent must refuse. "Returning 'looks
good'" is the canonical one for review agents. Each role
has its own specific anti-patterns — visual-judgement roles
must refuse normalisation; code-attackers must refuse
style-nits-as-bugs; etc.

## When to invoke
Concrete triggers. "Before closing a research plan" or "on
any diff that touches `pathtrace::integrator`" — not "when
useful."
```

## Skills × agents

A skill can invoke an agent as one of its steps. E.g. a
future `close-plan` skill could invoke `plan-skeptic` on the
plan-to-be-closed and refuse to mark it completed if the
skeptic surfaces a P1 attack the closer hasn't addressed.
This composition is the whole reason `skills/` and `agents/`
live side-by-side instead of in one directory.

## Currently scaffolded

| Role | Mandate | Invoke when |
|------|---------|-------------|
| [`research-critic.md`](research-critic.md) | Find published prior art that overlaps; find a baseline that beats the method; find an unstated weakness. | Research plan moves from `hypothesis` to `experimenting`, or any time before submission. |
| [`plan-skeptic.md`](plan-skeptic.md) | Find failure modes the plan doesn't address; find "Done when" criteria that don't actually deliver the goal; find missing cross-references. | Before closing an implementation plan; on any plan that's been `active` for >2 weeks. |
| [`code-attacker.md`](code-attacker.md) | Find concrete bugs, edge cases, race conditions, performance regressions, untested boundaries in a diff. | On plan-closing commits + on any refactor over ~100 lines. |
| [`code-defender.md`](code-defender.md) | Read the attacker's report; accept the real ones, refute misunderstandings with reasoning, defer with rationale. | Paired with `code-attacker`; never invoked alone. |
| [`render-attacker.md`](render-attacker.md) | Find visual regressions between two reference renders — lost detail, introduced artifacts, color shifts, halos, banding. | When re-rendering hero gallery references or comparing denoiser variants. |
| [`render-defender.md`](render-defender.md) | Find visual improvements in the new render. | Paired with `render-attacker`; never invoked alone. |
