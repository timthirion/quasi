---
name: pre-flight
description: Run the full quality-gate sequence (fmt-check, clippy -D warnings, wasm32 check, all-targets tests, Python tests if scripts/ touched). Auto-fix fmt drift once; surface other failures verbatim.
version: 0.1.0
---

# Pre-flight quality gates

The full sequence required before any commit + push. Codifies
the [[feedback_verify_ci_after_push]] memory entry: every CI
red I've ever landed on this repo came from skipping one of
these four steps locally.

## The sequence

Run these four, **in order**, halting on the first failure:

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo check --target wasm32-unknown-unknown --lib`
4. `cargo test --all-targets`

If the working tree touches `scripts/` (any file under that
directory, including additions), also run:

5. `python3 -m unittest discover scripts -p 'test_*.py' -v`

## Failure handling

### Step 1 (fmt) — auto-fix once, then re-check

`cargo fmt --check` failing means rustfmt would reformat
something. The right move is to run `cargo fmt --all`, then
re-run `cargo fmt --all -- --check` to verify it's now clean.
**Do this exactly once.** If `--check` still fails after the
auto-format, something is structurally wrong (likely an
`rustfmt::skip` directive on a file that doesn't satisfy the
rule even formatted); surface the diff verbatim and stop.

### Step 2 (clippy) — surface verbatim, never auto-fix

Clippy failures are code issues, not style drift. Do **not**
run `cargo clippy --fix`. Surface the error verbatim, including
the `file:line` and the suggested fix the lint emits. The
human (or the agent re-entering) decides whether to apply the
suggestion or rewrite the underlying code.

A subtle gotcha: clippy errors get truncated by the auto-fmt
re-runs in earlier steps if pre-flight is invoked inside a
pipeline that swallows exit codes. Always check that pre-flight
exits non-zero on clippy failure.

### Step 3 (wasm32) — distinguish target-missing from real error

If the error message contains "target `wasm32-unknown-unknown`
may not be installed" or similar, the fix is `rustup target add
wasm32-unknown-unknown` — not a code change. Surface that as a
setup issue, not a build failure.

For any other wasm32 error, surface verbatim. The most common
real failure here is a `cfg(not(target_arch = "wasm32"))`
boundary getting missed when a native-only crate (e.g. `image`
with the `hdr` feature) creeps into the wasm-visible code path.

### Step 4 (test) — surface verbatim with the failure summary

`cargo test --all-targets` failures need the test name + the
assertion message + the location, all of which the default
output includes. Surface the FAILED lines + the panic message
for each failing test. Don't truncate.

### Step 5 (Python, conditional) — surface verbatim

Same shape as test failures. The Python suite is small enough
that the default `unittest` output is fine to surface in full.

## Output shape

Report a per-step ✓ / ✗ summary. On ✗, include the failing
step's full output. On ✓ across all steps, a one-line
"pre-flight green" is enough — silence on success is OK but
explicit confirmation builds trust.

```
Pre-flight on <branch>:
  ✓ fmt
  ✓ clippy
  ✓ wasm32 check
  ✓ tests (118 passing across 16 suites)
  ✓ python tests (skipped — no scripts/ changes)
Pre-flight green.
```

## What pre-flight is NOT

* Not a commit step. Pre-flight gates *whether* to commit; it
  doesn't commit. A future `commit-and-push` skill would call
  pre-flight as its first action.
* Not a "fix things" loop. The fmt auto-fix is the **only**
  auto-correction. Everything else is surface-and-stop.
* Not a substitute for CI. CI runs these same checks on a
  clean Linux runner; pre-flight catches issues before the
  push so CI stays green by default, not after the fact.

## When to invoke

Before any `git commit`. Before any `git push` that follows a
commit you didn't pre-flight (e.g. a rebase, a merge, an
amend). After any non-trivial refactor, even if no commit is
planned yet — catches drift early.

Skip is fine for trivial commits (a README typo on its own
commit, a single comment fix) — but the moment the change
touches code, pre-flight should run. The cost is ~5 seconds on
a warm cache; the cost of a CI red is one extra commit + a
re-push + the next agent in the conversation seeing a yellow
dashboard.
