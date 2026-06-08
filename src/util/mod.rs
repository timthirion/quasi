//! Cross-cutting utilities that don't belong to a specific
//! pipeline. Currently houses the generic [`progress`] sink so the
//! path tracer's offscreen render — and any future long-running
//! task (BVH build, scene fetch, denoise pass) — can report
//! progress through one trait without forming a runtime dependency
//! on a specific bar implementation.

pub mod progress;
