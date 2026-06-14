//! Compute-dispatch recorder shared by the GPU solvers.
//!
//! Both solvers issue many compute dispatches per frame into a single encoder + single
//! `queue.submit`. The naive form opens one `begin_compute_pass`/`end` pair **per dispatch**;
//! in the browser each pair costs ~4µs of CPU pass-bookkeeping plus a wasm→JS boundary
//! crossing, so a ~500-dispatch frame burns ~2ms of pure overhead.
//!
//! This recorder batches **consecutive** dispatches into ONE compute pass when per-pass
//! timestamp profiling is OFF (the web/production path, and any device without the
//! `TIMESTAMP_QUERY` feature). WebGPU/Dawn automatically inserts the storage-buffer
//! read-after-write memory barriers between dispatches in the same pass, so dependent solver
//! stages produce identical results — only the CPU begin/end overhead is dropped. The
//! equivalence is proven by `tests/batch_pass_equivalence.rs`.
//!
//! When timestamps ARE available (native profiling / the perf tests / `examples/`), the
//! recorder keeps the legacy per-pass form so `ComputePassTimestampWrites` still gives the
//! profiler a per-pass breakdown. The two modes record byte-identical dispatch *work*; they
//! differ only in how that work is partitioned into passes.
//!
//! `dispatches_per_frame` counts **dispatches**, not passes, in both modes.
//!
//! ## Encoder hazard
//! A non-dispatch encoder op (`copy_buffer_to_buffer`, `resolve_query_set`, …) cannot run
//! while a compute pass is open. In batch mode the recorder lazily holds an open pass, so the
//! caller MUST call [`PassRecorder::flush`] before touching the encoder directly. After the
//! last dispatch, call [`PassRecorder::finish`] to close any open pass and harvest the
//! bookkeeping. (`forget_lifetime` is used so the open pass can be stored across dispatches;
//! correctness of the flush discipline is enforced by wgpu/Dawn validation at submit, and by
//! the equivalence gate.)

/// Per-pass timestamp wiring borrowed for the lifetime of a frame's recording. The query set
/// is written two indices per dispatch (begin/end); `capacity` caps how many fit.
pub struct TimestampSink<'a> {
    pub qset: &'a wgpu::QuerySet,
    pub capacity: u32,
}

/// Records dispatches into an encoder, batching them into one pass when timestamps are off.
pub struct PassRecorder<'a> {
    /// `Some` ⇒ per-pass timestamped mode (native profiling). `None` ⇒ batched single-pass mode.
    ts: Option<TimestampSink<'a>>,
    /// The currently-open batched pass (batch mode only); `None` between flushes.
    open: Option<wgpu::ComputePass<'static>>,
    /// Next free timestamp query index (advances by 2 per timestamped dispatch).
    cursor: u32,
    /// Per-dispatch labels, parallel to the timestamp pairs (timestamp mode only).
    labels: Vec<String>,
    /// Total dispatches recorded this frame (both modes).
    dispatches: u32,
}

impl<'a> PassRecorder<'a> {
    /// Build a recorder. Pass `Some(sink)` for native per-pass timestamp profiling, `None`
    /// for the batched single-pass path (web / no timestamp feature).
    pub fn new(ts: Option<TimestampSink<'a>>) -> Self {
        Self {
            ts,
            open: None,
            cursor: 0,
            labels: Vec::new(),
            dispatches: 0,
        }
    }

    /// Record one compute dispatch. In timestamp mode this opens its own pass with begin/end
    /// timestamp writes (legacy behavior). In batch mode it appends to the open pass, opening
    /// a fresh one if none is live.
    pub fn dispatch(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        label: &str,
        groups: u32,
    ) {
        self.dispatches += 1;
        match &self.ts {
            Some(sink) => {
                // Per-pass timestamped pass (one begin/end pair per dispatch).
                let tw = if self.cursor + 1 < sink.capacity {
                    let b = self.cursor;
                    let e = self.cursor + 1;
                    self.cursor += 2;
                    self.labels.push(label.to_string());
                    Some(wgpu::ComputePassTimestampWrites {
                        query_set: sink.qset,
                        beginning_of_pass_write_index: Some(b),
                        end_of_pass_write_index: Some(e),
                    })
                } else {
                    None
                };
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: tw,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(bind_group), &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            None => {
                // Batched: reuse the open pass; lazily open one (no timestamp writes).
                if self.open.is_none() {
                    let pass = enc
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("batched-compute"),
                            timestamp_writes: None,
                        })
                        .forget_lifetime();
                    self.open = Some(pass);
                }
                let pass = self.open.as_mut().expect("pass just opened");
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(bind_group), &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    /// Close any open batched pass so the encoder is free for a direct op
    /// (`copy_buffer_to_buffer`, `resolve_query_set`). No-op in timestamp mode (passes are
    /// already closed per-dispatch) and when no pass is open.
    pub fn flush(&mut self) {
        // Dropping the pass ends it.
        self.open = None;
    }

    /// Drop into batched mode for the remaining dispatches (used by the multi-substep solver to
    /// timestamp only substep 0, then batch the rest). After this call the recorded `cursor` /
    /// `labels` stop growing, matching the legacy "timestamps on substep 0 only" contract that
    /// the perf test / scaling example depend on (`total_micros()` = one substep's sum).
    pub fn disable_timestamps(&mut self) {
        self.ts = None;
    }

    /// Close any open pass and return `(dispatches, cursor, labels)` for the solver's
    /// bookkeeping. `cursor`/`labels` are meaningful only in timestamp mode (empty otherwise).
    pub fn finish(mut self) -> (u32, u32, Vec<String>) {
        self.flush();
        (self.dispatches, self.cursor, self.labels)
    }
}
