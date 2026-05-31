use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PassTimings {
    pub emit_predict_ms: f32,
    pub hash_ms: f32,
    pub constraints_ms: f32,
    pub boundaries_ms: f32,
    pub extraction_ms: f32,
    pub render_pack_ms: f32,
    pub metrics_ms: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct Profiler {
    samples: VecDeque<f32>,
    max_samples: usize,
    last_frame_ms: f32,
    last_passes: PassTimings,
    timestamp_queries_available: bool,
}

impl Default for Profiler {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(120),
            max_samples: 120,
            last_frame_ms: 0.0,
            last_passes: PassTimings::default(),
            timestamp_queries_available: false,
        }
    }
}

impl Profiler {
    pub(crate) fn record_frame(&mut self, frame_ms: f32, passes: PassTimings) {
        self.last_frame_ms = frame_ms;
        self.last_passes = passes;
        if self.samples.len() == self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(frame_ms);
    }

    pub(crate) fn average_ms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f32>() / self.samples.len() as f32
    }

    pub(crate) fn p95_ms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut values: Vec<_> = self.samples.iter().copied().collect();
        values.sort_by(|a, b| a.total_cmp(b));
        values[((values.len() - 1) as f32 * 0.95).round() as usize]
    }

    pub(crate) fn max_ms(&self) -> f32 {
        self.samples.iter().copied().fold(0.0, f32::max)
    }

    pub(crate) fn last_passes(&self) -> PassTimings {
        self.last_passes
    }

    pub(crate) fn timestamp_queries_available(&self) -> bool {
        self.timestamp_queries_available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_records_frame_statistics_without_gpu_timestamps() {
        let mut profiler = Profiler::default();
        profiler.record_frame(12.0, PassTimings::default());
        profiler.record_frame(18.0, PassTimings::default());
        assert!(profiler.average_ms() >= 15.0);
        assert!(profiler.p95_ms() >= 12.0);
        assert_eq!(profiler.max_ms(), 18.0);
        assert!(!profiler.timestamp_queries_available());
    }
}
