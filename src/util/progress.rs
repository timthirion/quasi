//! Generic progress reporting for long-running CLI tasks.
//!
//! [`ProgressSink`] is a trivial sink trait so callers can plug in
//! anything from a real terminal bar to a noop. The path tracer's
//! offscreen render is the first consumer, but future callers
//! (BVH build, scene fetch, denoise, EXR encode) plug into the same
//! trait without forming a runtime dependency on a specific bar.
//!
//! The shipped [`Bar`] implementation writes a single-line
//! carriage-return progress display to stderr with elapsed + ETA
//! estimates. It throttles redraws to ~5 Hz so noisy callers don't
//! flood the terminal.

use std::io::{self, Write};
use std::time::{Duration, Instant};

/// A sink to which a long-running task reports `(done, total)` ticks.
/// Implementations decide what to do with them — write to stderr,
/// push to a log, no-op, etc. `Send` is required so callers can move
/// sinks across thread boundaries (e.g. into an async render driver).
pub trait ProgressSink: Send {
    /// Report progress. `done` is the number of units completed
    /// so far (e.g. samples per pixel rendered), `total` is the
    /// expected final count. Callers may call this very frequently;
    /// implementations should throttle their own output.
    fn tick(&mut self, done: u64, total: u64);

    /// Mark the task complete. Implementations should finalise any
    /// transient display (e.g. emit a trailing newline so the next
    /// log line starts on its own row).
    fn finish(&mut self);
}

/// Noop sink — drops all ticks. Use when the caller has no
/// terminal (CI, tests, batch jobs) or progress is otherwise not
/// wanted.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn tick(&mut self, _done: u64, _total: u64) {}
    fn finish(&mut self) {}
}

/// Stderr progress bar with elapsed + ETA + rate. Single line,
/// carriage-return redraw, throttled to roughly 5 Hz so that very
/// fast callers don't bog the terminal. Always falls back to
/// "tick every change" once `done == total` is reached so the
/// final state is always rendered.
pub struct Bar {
    label: String,
    units: &'static str,
    start: Instant,
    last_draw: Instant,
    redraw_every: Duration,
    finished: bool,
}

impl Bar {
    /// New bar. `label` prefixes each line (e.g. `"render"`),
    /// `units` is the suffix on the count + rate (e.g. `"spp"`).
    pub fn new(label: impl Into<String>, units: &'static str) -> Self {
        let now = Instant::now();
        Self {
            label: label.into(),
            units,
            start: now,
            // Force the first tick to draw by parking last_draw a
            // full redraw interval into the past.
            last_draw: now - Duration::from_secs(60),
            redraw_every: Duration::from_millis(200),
            finished: false,
        }
    }
}

impl ProgressSink for Bar {
    fn tick(&mut self, done: u64, total: u64) {
        if self.finished {
            return;
        }
        let now = Instant::now();
        let force = done >= total;
        if !force && now.duration_since(self.last_draw) < self.redraw_every {
            return;
        }
        self.last_draw = now;
        let elapsed = now.duration_since(self.start);
        let frac = if total > 0 {
            (done as f64 / total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let bar = render_bar(frac, BAR_WIDTH);
        let rate = if elapsed.as_secs_f64() > 0.0 {
            done as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let eta = if rate > 0.0 && done < total {
            Some(Duration::from_secs_f64((total - done) as f64 / rate))
        } else {
            None
        };
        let eta_str = eta.map(format_duration).unwrap_or_else(|| "—".to_string());
        let _ = write!(
            io::stderr(),
            "\r{} {} {:>3.0}%  {}/{} {}  {:.1} {}/s  eta {}     ",
            self.label,
            bar,
            frac * 100.0,
            done,
            total,
            self.units,
            rate,
            self.units,
            eta_str
        );
        let _ = io::stderr().flush();
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let elapsed = Instant::now().duration_since(self.start);
        let _ = writeln!(
            io::stderr(),
            "\r{} done in {}{:width$}",
            self.label,
            format_duration(elapsed),
            "",
            width = TRAILING_CLEAR_WIDTH
        );
        let _ = io::stderr().flush();
    }
}

const BAR_WIDTH: usize = 28;
/// Wide enough to overwrite the longest in-progress line so the
/// finish line doesn't leave residue from the redraw.
const TRAILING_CLEAR_WIDTH: usize = 80;

fn render_bar(frac: f64, width: usize) -> String {
    let filled = (frac * width as f64).round() as usize;
    let filled = filled.min(width);
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for _ in 0..filled {
        s.push('█');
    }
    for _ in filled..width {
        s.push('░');
    }
    s.push(']');
    s
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        // Sub-minute: include tenths so short renders still feel
        // responsive ("12.4s" not "12s").
        let frac = d.as_secs_f64() - secs as f64;
        format!("{}.{:01}s", s, (frac * 10.0).floor() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_compiles_and_does_nothing() {
        let mut s = NullSink;
        s.tick(0, 100);
        s.tick(50, 100);
        s.tick(100, 100);
        s.finish();
    }

    #[test]
    fn render_bar_empty() {
        assert_eq!(render_bar(0.0, 4), "[░░░░]");
    }

    #[test]
    fn render_bar_half() {
        assert_eq!(render_bar(0.5, 4), "[██░░]");
    }

    #[test]
    fn render_bar_full() {
        assert_eq!(render_bar(1.0, 4), "[████]");
    }

    #[test]
    fn render_bar_clamps_over_one() {
        // Caller passed >1 by accident — we shouldn't grow the bar
        // past `width`.
        assert_eq!(render_bar(1.5, 4), "[████]");
    }

    #[test]
    fn format_duration_subminute() {
        assert_eq!(format_duration(Duration::from_millis(12_400)), "12.4s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(Duration::from_secs(125)), "2m05s");
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h01m01s");
    }

    #[test]
    fn bar_finish_is_idempotent() {
        let mut b = Bar::new("test", "spp");
        b.tick(5, 10);
        b.finish();
        b.finish(); // second call is a noop, must not panic.
    }

    #[test]
    fn bar_tick_after_finish_is_noop() {
        let mut b = Bar::new("test", "spp");
        b.finish();
        b.tick(99, 100); // must not redraw / panic.
    }

    #[test]
    fn bar_object_is_send() {
        // Compile-time check: ProgressSink: Send is load-bearing
        // so callers can move a Bar into a worker thread.
        fn assert_send<T: Send>() {}
        assert_send::<Bar>();
        assert_send::<NullSink>();
        assert_send::<Box<dyn ProgressSink>>();
    }
}
