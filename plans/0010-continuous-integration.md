# Continuous integration

- **Status:** done
- **Last updated:** 2026-06-05
- **Last touched on:** CI-base landed — plan closed

## Goal

Stand up GitHub Actions CI that runs the full quality gate on
every push and PR. Output: a green badge for the README, plus the
peace of mind that nothing lands broken. Once CI is green, a
one-liner adds the badge to `README.md` ahead of the aesthetic
rewrite (sequenced next as plan 0011 or inline).

## Context

What's already in:

- 159 native tests across `cargo test` (98 lib + 61 integration
  in `tests/`).
- 10 Python tests under `scripts/` for the `.qvg` writer.
- `cargo fmt` defaults; `cargo clippy --all-targets -- -D warnings`
  expected clean (this is doctrine in `AGENTS.md`).
- `cargo check --target wasm32-unknown-unknown` for the
  native-and-web-in-lockstep rule.
- GPU-dependent tests are already `#[ignore]`'d so headless Linux
  runners don't need a display server.

What this plan is **not**:

- A self-hosted runner with a real GPU. The `#[ignore]`'d GPU
  tests stay opt-in; CI runs the CPU-side quality gate.
- Release automation, version bumping, or publishing artefacts.
  Those land if/when the project ships releases.
- A wasm-pack build matrix or browser smoke test. Just
  `cargo check` for the wasm target so we know it compiles.

## Design

### Workflow file: `.github/workflows/ci.yml`

Single workflow, triggered on `push` and `pull_request` against any
branch. Two jobs that can run in parallel:

```yaml
jobs:
  rust:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - install stable toolchain + rustfmt + clippy
      - install wasm32-unknown-unknown target
      - cache cargo registry + target dir
      - cargo fmt --all -- --check
      - cargo clippy --all-targets -- -D warnings
      - cargo test --all-targets
      - cargo check --target wasm32-unknown-unknown

  python:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - install Python 3.11
      - python3 -m unittest discover scripts -p 'test_*.py' -v
```

Both jobs are required for the green badge. The Rust job carries
all the meat; the Python job is fast and pins the `.qvg` writer.

### Caching

Use `actions/cache` keyed on `Cargo.lock` + the rustc commit hash
so a re-run after a non-dependency change is fast. ~3-minute cold
build, ~30-second warm rebuild is typical.

### Badge

Once the workflow has run at least once:

```markdown
![CI](https://github.com/timthirion/quasi/actions/workflows/ci.yml/badge.svg)
```

Goes at the top of `README.md` alongside the other shield badges
in plan 0011.

## Milestones

### CI-base ✅
Single milestone covering the whole thing.

- [x] `.github/workflows/ci.yml`: Rust job runs fmt + clippy + test
      + wasm32 check; Python job runs the `scripts/` test discovery.
      Concurrency group cancels in-flight runs when a newer commit
      lands. `RUSTFLAGS: -D warnings` makes accidental warnings
      hard failures.
- [x] Caching via `Swatinem/rust-cache@v2` keyed on `Cargo.lock`.
      No apt deps needed — `cargo check --target
      wasm32-unknown-unknown --lib` builds cleanly without the
      native winit toolchain.
- [x] First push (commit `40aec02`) triggered the workflow; both
      jobs landed green on the first attempt. Rust job: 2 m 35 s
      cold. Python job: ~10 s.
- [x] CI status badge added to `README.md` alongside license + Rust
      edition + WebGPU shields, ahead of the aesthetic README
      rewrite in plan 0011.

**Pre-existing nits fixed in the same commit.** `cargo fmt` had
drifted on 12 files; `cargo clippy --all-targets -- -D warnings`
flagged four kinds of leftovers (excessive-precision golden-ratio
literal, manual range check, unused loop index). All cleaned so
the first CI run was fully green rather than reporting a backlog.

**Node 20 deprecation warning.** `actions/checkout@v4` +
`actions/setup-python@v5` print a Node 20 deprecation notice; June
2026 cut-over to Node 24. Tracking as a follow-up — bump to
`actions/checkout@v5` (when published) or set
`FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true`. Not blocking.

## Open questions

- **`cargo test --all-targets` vs `cargo test`.** `--all-targets`
  builds and runs examples + doctests too. Default for now;
  drop back if examples cost too much CI time.
- **wgpu in CI?** wgpu pulls a lot of platform deps even for
  `cargo check`. If the wasm32 check is fast we leave it; if
  it's slow + fragile we drop it and rely on local discipline.
- **Python version pin.** 3.11 is the LTS-ish default. Bump if
  the workflow needs newer.

## Done when

- A push to `main` (or any PR) triggers both jobs.
- Both jobs land green from a clean clone.
- The README has a CI status badge linked to the workflow.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  and Python unittests all stay green at HEAD.
