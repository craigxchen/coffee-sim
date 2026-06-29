//! Pour recipes: time-varying water-addition patterns, ported from the v1 `pour.rs`.
//!
//! A [`PourScript`] is a sequence of [`PourCommand`]s, each a time window with a flow rate and a
//! spatial pattern. Sampling at time `t` yields the current pour position (normalized to `[-1, 1]`
//! relative to the bed center) and flow rate (mL/s). A gap between commands (or before/after the
//! script) is a wait / drawdown phase: no pour. The driver maps the normalized position to a world
//! kettle position (via the bed radius) and converts mL/s to the solver's volumetric `flow_rate`.

use std::f32::consts::PI;

/// Spatial pour pattern (positions normalized to `[-1, 1]` about the bed center).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PourPattern {
    /// Pour at the center of the bed.
    Center,
    /// Spiral pour: the stream sweeps its radius between `r_min` and `r_max` while rotating.
    ///
    /// Note: the radius triangle-wave runs at the same `freq_hz` as the angular rotation, so the
    /// trace is a rosette (petals), not a monotonically-expanding spiral. Kept faithful to the v1
    /// recipe; it reads as a believable agitating pour and the pattern stays within `[r_min, r_max]`.
    Spiral {
        freq_hz: f32,
        r_min: f32,
        r_max: f32,
    },
    /// Ring pour at a fixed radius (normalized), slowly rotating.
    Ring { radius: f32 },
    /// Pour at a fixed point (normalized `[-1, 1]`).
    Point { x: f32, y: f32 },
}

/// A single timed pour command.
#[derive(Clone, Copy, Debug)]
pub struct PourCommand {
    /// Start time (seconds, inclusive).
    pub t_start: f32,
    /// End time (seconds, exclusive).
    pub t_end: f32,
    /// Flow rate (mL/s).
    pub flow_rate: f32,
    /// Spatial pattern for this phase.
    pub pattern: PourPattern,
}

/// A complete pour recipe: a sequence of commands sampled over the brew.
#[derive(Clone, Debug, Default)]
pub struct PourScript {
    pub commands: Vec<PourCommand>,
}

impl PourScript {
    /// Sample the script at time `t`: returns `(x, y, flow_rate_ml_s)` with `(x, y)` normalized to
    /// `[-1, 1]`. Returns `(0, 0, 0)` when no command is active (wait / drawdown).
    pub fn sample(&self, t: f32) -> (f32, f32, f32) {
        for cmd in &self.commands {
            if t >= cmd.t_start && t < cmd.t_end {
                let (x, y) = sample_pattern(&cmd.pattern, t);
                return (x, y, cmd.flow_rate);
            }
        }
        (0.0, 0.0, 0.0)
    }

    /// Total script duration (end time of the last command).
    pub fn total_duration(&self) -> f32 {
        self.commands.iter().map(|c| c.t_end).fold(0.0, f32::max)
    }

    /// Total water dispensed by the script (mL) = Σ flow_rate · window duration.
    pub fn total_water_ml(&self) -> f32 {
        self.commands
            .iter()
            .map(|c| c.flow_rate * (c.t_end - c.t_start))
            .sum()
    }
}

/// `(x, y)` in `[-1, 1]` for a pattern at time `t`.
fn sample_pattern(pattern: &PourPattern, t: f32) -> (f32, f32) {
    match *pattern {
        PourPattern::Center => (0.0, 0.0),
        PourPattern::Spiral {
            freq_hz,
            r_min,
            r_max,
        } => {
            if freq_hz <= 0.0 {
                return (0.0, 0.0);
            }
            let angle = 2.0 * PI * freq_hz * t;
            // Radius triangle-wave over one period (sweeps r_min→r_max→r_min).
            let period = 1.0 / freq_hz;
            let phase = (t % period) / period; // 0..1
            let r = if phase < 0.5 {
                r_min + (r_max - r_min) * (phase * 2.0)
            } else {
                r_max - (r_max - r_min) * ((phase - 0.5) * 2.0)
            };
            (r * angle.cos(), r * angle.sin())
        }
        PourPattern::Ring { radius } => {
            let angle = 2.0 * PI * 0.5 * t; // ~0.5 Hz rotation
            (radius * angle.cos(), radius * angle.sin())
        }
        PourPattern::Point { x, y } => (x, y),
    }
}

/// Classic spiral pour: bloom center 0–10 s, wait 10–40 s, spiral 40–130 s, then drawdown.
pub fn classic_spiral() -> PourScript {
    PourScript {
        commands: vec![
            PourCommand {
                t_start: 0.0,
                t_end: 10.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            PourCommand {
                t_start: 40.0,
                t_end: 130.0,
                flow_rate: 3.0,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.4,
                    r_min: 0.15,
                    r_max: 0.75,
                },
            },
        ],
    }
}

/// Center-only pour: bloom center 0–10 s, wait 10–40 s, center 40–130 s, then drawdown.
pub fn center_only() -> PourScript {
    PourScript {
        commands: vec![
            PourCommand {
                t_start: 0.0,
                t_end: 10.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            PourCommand {
                t_start: 40.0,
                t_end: 130.0,
                flow_rate: 3.0,
                pattern: PourPattern::Center,
            },
        ],
    }
}

/// Pulse pour: bloom 0–8 s, then spiral pulses separated by waits (drawdown between pulses).
pub fn pulse_pour() -> PourScript {
    let pulse = |t_start: f32| PourCommand {
        t_start,
        t_end: t_start + 15.0,
        flow_rate: 5.0,
        pattern: PourPattern::Spiral {
            freq_hz: 0.3,
            r_min: 0.1,
            r_max: 0.6,
        },
    };
    PourScript {
        commands: vec![
            PourCommand {
                t_start: 0.0,
                t_end: 8.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            pulse(25.0),
            pulse(50.0),
            pulse(75.0),
            pulse(100.0),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_water_is_sum_of_windows() {
        let s = classic_spiral();
        // 10 s · 5 mL/s + 90 s · 3 mL/s = 50 + 270 = 320 mL.
        assert!((s.total_water_ml() - 320.0).abs() < 1.0e-3);
        assert!((s.total_duration() - 130.0).abs() < 1.0e-3);
    }

    #[test]
    fn sample_returns_active_command_and_gaps_are_dry() {
        let s = classic_spiral();
        // Bloom (center, 5 mL/s).
        let (x, y, f) = s.sample(5.0);
        assert_eq!((x, y), (0.0, 0.0));
        assert!((f - 5.0).abs() < 1.0e-6);
        // Wait gap 10–40 s → no pour.
        assert_eq!(s.sample(25.0), (0.0, 0.0, 0.0));
        // Main spiral → 3 mL/s.
        assert!((s.sample(80.0).2 - 3.0).abs() < 1.0e-6);
        // Drawdown after the script → no pour.
        assert_eq!(s.sample(200.0), (0.0, 0.0, 0.0));
    }

    #[test]
    fn patterns_are_bounded_and_correct() {
        assert_eq!(sample_pattern(&PourPattern::Center, 3.3), (0.0, 0.0));
        assert_eq!(
            sample_pattern(&PourPattern::Point { x: 0.4, y: -0.2 }, 7.0),
            (0.4, -0.2)
        );
        // Ring stays on its radius.
        let (rx, ry) = sample_pattern(&PourPattern::Ring { radius: 0.6 }, 1.7);
        assert!(((rx * rx + ry * ry).sqrt() - 0.6).abs() < 1.0e-4);
        // Spiral stays within [r_min, r_max].
        let pat = PourPattern::Spiral {
            freq_hz: 0.4,
            r_min: 0.15,
            r_max: 0.75,
        };
        for i in 0..200 {
            let t = i as f32 * 0.13;
            let (x, y) = sample_pattern(&pat, t);
            let r = (x * x + y * y).sqrt();
            assert!(
                (0.15 - 1.0e-4..=0.75 + 1.0e-4).contains(&r),
                "spiral radius {r} out of [r_min, r_max] at t={t}"
            );
        }
    }

    #[test]
    fn sample_is_deterministic() {
        let s = classic_spiral();
        assert_eq!(s.sample(80.0), s.sample(80.0));
    }
}
