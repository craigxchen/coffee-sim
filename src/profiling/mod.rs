//! Local GPU profiler (deliberately thin for now).
//!
//! Reports two things into [`Profile`]: per-pass GPU timings and — always —
//! `dispatches_per_frame`, the single variable that predicts the eventual browser/mobile
//! cost gap. We keep the dispatch counter even though it's trivial now: it makes the later
//! browser pass a quick verification rather than a redo, and it's directly actionable
//! (batch a solver's iteration passes into fewer dispatches).
//!
//! Per-pass `timestamp-query` capture (QuerySet write/resolve/readback of the *previous*
//! frame, to avoid a sync stall) is wired alongside the first real compute pass; Phase 0
//! has no passes, so `passes` is empty by construction. When `TIMESTAMP_QUERY` is absent
//! the timing path is simply skipped — dispatches are still counted (first-class fallback).

/// Per-frame profile snapshot. Read-only output consumed by `ui` and comparison harnesses.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    /// `(pass label, GPU duration µs)`. Empty when there are no passes or timestamps are
    /// unavailable.
    pub passes: Vec<(String, f32)>,
    /// GPU compute dispatches issued this frame.
    pub dispatches_per_frame: u32,
}

impl Profile {
    /// Total measured GPU time across passes (µs).
    pub fn total_micros(&self) -> f32 {
        self.passes.iter().map(|(_, us)| *us).sum()
    }
}

/// Accumulates per-frame profiling data. A solver owns one and drives it during `step`,
/// returning its `snapshot()` from `Solver::profile`.
pub struct Profiler {
    timestamps_supported: bool,
    dispatches: u32,
    passes: Vec<(String, f32)>,
}

impl Profiler {
    pub fn new(timestamps_supported: bool) -> Self {
        Self {
            timestamps_supported,
            dispatches: 0,
            passes: Vec::new(),
        }
    }

    /// Whether per-pass `timestamp-query` timing is available on this device. The
    /// timestamp-resolve helper consults this before emitting pass timings.
    pub fn timestamps_supported(&self) -> bool {
        self.timestamps_supported
    }

    /// Reset per-frame counters at the start of a frame.
    pub fn begin_frame(&mut self) {
        self.dispatches = 0;
        self.passes.clear();
    }

    /// Count one GPU compute dispatch.
    pub fn record_dispatch(&mut self) {
        self.dispatches += 1;
    }

    /// Record a resolved per-pass GPU duration (µs). Fed by the timestamp-resolve helper
    /// once a solver has compute passes; no-op solvers never call it.
    pub fn record_pass(&mut self, label: impl Into<String>, micros: f32) {
        self.passes.push((label.into(), micros));
    }

    /// Snapshot the current frame's profile.
    pub fn snapshot(&self) -> Profile {
        Profile {
            passes: self.passes.clone(),
            dispatches_per_frame: self.dispatches,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_dispatches_per_frame() {
        let mut p = Profiler::new(false);
        p.begin_frame();
        for _ in 0..7 {
            p.record_dispatch();
        }
        assert_eq!(p.snapshot().dispatches_per_frame, 7);

        // Counter resets each frame.
        p.begin_frame();
        assert_eq!(p.snapshot().dispatches_per_frame, 0);
    }

    #[test]
    fn no_panic_without_timestamps_and_empty_passes() {
        // The fallback path (no TIMESTAMP_QUERY) still produces a valid, empty profile.
        let mut p = Profiler::new(false);
        p.begin_frame();
        let snap = p.snapshot();
        assert!(snap.passes.is_empty());
        assert_eq!(snap.total_micros(), 0.0);
        assert!(!p.timestamps_supported());
    }

    #[test]
    fn records_pass_timings() {
        let mut p = Profiler::new(true);
        p.begin_frame();
        p.record_pass("p2g", 12.5);
        p.record_pass("g2p", 7.5);
        let snap = p.snapshot();
        assert_eq!(snap.passes.len(), 2);
        assert_eq!(snap.total_micros(), 20.0);
    }
}
