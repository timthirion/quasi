---
name: commit-and-push
description: Run pre-flight, construct a project-convention commit message, commit, push to origin, watch CI on the just-pushed run, refuse to consider the push "done" until CI is green. Codifies [[feedback_verify_ci_after_push]] and the autonomy bargain from [[feedback_autonomy]].
---

# Commit + push

The closing half of any unit of work. Pre-flight gates
*whether* to commit; this skill takes a green pre-flight,
constructs the commit, pushes it, and stays attached until CI
confirms green. The autonomy bargain from
[[feedback_autonomy]] is: **commit and push freely, but CI must
stay green** — this skill enforces the second half mechanically.

## The sequence

1. **Invoke [`pre-flight`](pre-flight.md).** If it returns red,
   stop. Surface the pre-flight failure verbatim. **Do not
   commit.** The commit only happens with a green pre-flight.

2. **Stage the intended files.** `git add -A` if everything in
   the working tree should ship; `git add <paths>` when only a
   subset should. Prefer the explicit-paths form when the
   working tree has uncommitted experiments next to the
   intended change.

3. **Construct the commit message** per the project convention
   (below). Use a HEREDOC to pass the message so the body
   formatting survives intact. The subject is one line, ≤ 72
   chars, present tense. The body explains the **why**, not
   the **what** — git diff already shows the what.

4. **Commit.** `git commit -m "$(cat <<'EOF' ... EOF)"`. If
   the commit fails (pre-commit hook, missing sign-off), surface
   the error. Do **not** retry with `--no-verify`.

5. **Push to origin/main** (or the active branch). If push is
   rejected for non-fast-forward, **do not auto-force**.
   Surface the rejection — it likely means upstream moved and
   the human (or a `rebase-and-retry` skill) decides next.

6. **Watch CI on the new commit.** `gh run watch
   $(gh run list --workflow=CI --limit 1 --json databaseId
   --jq '.[0].databaseId') --exit-status`. Confirm the run is
   for the just-pushed SHA, not a stale older one.

7. **On CI green:** report the run ID + duration. The push is
   done.

8. **On CI red:** fetch the failing job's log
   (`gh run view <id> --log-failed`), surface the root error
   verbatim, and treat the push as **not done**. The autonomy
   bargain is broken if you walk away from red CI.

## Commit message convention

Subject (line 1): present-tense, ≤ 72 chars, no trailing
period. Typical shapes:

```
PT-many-lights: power-weighted emitter pick + 3-light showcase
Bunny UVs: cylindrical via morsel parameterize + storage-buf fix
Stop committing reference-render EXRs (-190 MB of clone weight)
```

Body (after blank line): why the change exists, what the
trade-offs are, what would have happened without it. Reference
plans (`plan 0019 PT-vertex-tangent`), commits (`commit 5bb1862`),
and memory entries (`[[feedback_verify_ci_after_push]]`) by
exact identifier. Wrap body at ~72 chars but don't be religious
about it — `cargo`, `wgpu`, and URL-shaped lines can spill.

Footer (after blank line): always include the Co-Authored-By
line that the autonomy memory codifies:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

The full pattern, ready to pipe:

```bash
git commit -m "$(cat <<'EOF'
Subject line ≤ 72 chars

Body paragraph explaining the why. Reference plans, commits,
and memory entries by exact identifier. Don't summarise the
diff; the diff is the diff.

Trade-offs paragraph (when there are any). What we chose, what
we explicitly didn't, what the failure mode would be if the
choice was wrong.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Failure handling

### Pre-flight red

Surface the pre-flight failure verbatim; do not commit; do not
attempt to fix the underlying issue silently. The caller (or
human) sees the failure, addresses it, and re-invokes
`commit-and-push`. Auto-fixing under the autonomy umbrella
hides the failure mode from the human's review.

### Push rejected

Most commonly: upstream moved and the push is a non-fast-forward.
Surface the error; do **not** force-push to recover. The
correct response is `git pull --rebase`, re-run pre-flight on
the rebased tree, then re-invoke. (A `rebase-and-retry` skill
would automate this; until it exists, hand the rejection back
to the caller.)

Force-push to main is **never** safe under autonomy mode. The
single exception is the explicit `git filter-repo` history
rewrite, which the user authorises explicitly and which is
not the domain of this skill.

### CI red post-push

Fetch `gh run view <id> --log-failed` and read the failing
step's tail. If the failure is fmt drift (pre-flight should
have caught it — investigate why it didn't), retry with
`cargo fmt` + a fresh `commit-and-push`. If the failure is
clippy, test, or wasm check, surface verbatim and stop —
**do not auto-commit a fix**. The next pre-flight + commit
should be deliberate, not reflexive.

### Push succeeded, CI still queued/in-progress

`gh run watch` blocks until the run completes. Don't exit
the skill early. The autonomy bargain isn't "push and trust
CI" — it's "push and confirm CI."

## Output shape

On full green:

```
commit-and-push on <branch>:
  ✓ pre-flight green
  ✓ committed <SHA> ("<subject>")
  ✓ pushed to origin/<branch>
  ✓ CI green (run <id>, <duration>)
Done.
```

On any red, surface the failure verbatim and stop. Don't
summarise — the caller needs the actual error to decide what
to do.

## What this is NOT

* Not a generic git wrapper. `git push` is one command; this
  skill earns its file by *bundling pre-flight + push + CI
  confirmation into the smallest unit of work the caller
  cares about*. Stripping any one step (skipping pre-flight,
  not waiting for CI) collapses the value.

* Not the place for `--amend`. Amending a pushed commit
  requires force-push, which this skill won't do. If the
  intent is amend-and-re-push, that's a separate skill or a
  human decision.

* Not the place for branch creation. `git checkout -b` and
  `git push -u origin <branch>` are upstream of this skill.

* Not the place for tagging / releasing. Tagged releases are
  rare enough to be a human decision.

## When to invoke

Whenever a logical unit of work is complete and the working
tree carries the change. Common callers:

* A milestone-closing edit-and-test loop ("PT-vertex-tangent
  landed, push it").
* A skill that produces side effects worth committing (a
  future `bake-assets` skill might invoke `commit-and-push`
  when scenes drift).
* A `close-plan` invocation that needs to land the plan-status
  edit + the closing notes.

Don't invoke after every single file write. The unit of work
should be a coherent commit — the agent's discretion. The
green CI confirmation is the proof that the commit was the
right unit.
