//! Pour script system for modeling time-varying water addition patterns.
//!
//! A `PourScript` consists of a sequence of `PourCommand` entries, each
//! defining a time window, flow rate, and spatial pour pattern. The script
//! is sampled at any time `t` to obtain the current pour position and
//! flow rate.

use std::f64::consts::PI;

/// Spatial pour pattern.
#[derive(Clone, Debug)]
pub enum PourPattern {
    /// Pour at the center of the bed.
    Center,
    /// Spiral pour: the stream traces a spiral between r_min and r_max.
    Spiral {
        freq_hz: f64,
        r_min: f64,
        r_max: f64,
    },
    /// Ring pour at a fixed radius (normalized to bed radius).
    Ring { radius: f64 },
    /// Pour at a specific (x, y) point, normalized to [-1, 1].
    Point { x: f64, y: f64 },
}

/// A single pour command within a script.
#[derive(Clone, Debug)]
pub struct PourCommand {
    /// Start time (seconds).
    pub t_start: f64,
    /// End time (seconds).
    pub t_end: f64,
    /// Flow rate in mL/s.
    pub flow_rate: f64,
    /// Spatial pattern for this pour phase.
    pub pattern: PourPattern,
}

/// A sequence of pour commands that defines the complete pour recipe.
#[derive(Clone, Debug)]
pub struct PourScript {
    pub commands: Vec<PourCommand>,
}

impl PourScript {
    /// Sample the pour script at time `t`.
    ///
    /// Returns `(pour_x, pour_y, flow_rate_ml_s)` where coordinates are
    /// normalized to [-1, 1] relative to bed center. If no command is active
    /// at time `t`, returns `(0.0, 0.0, 0.0)` (no pour / wait / drawdown).
    pub fn sample(&self, t: f64) -> (f64, f64, f64) {
        for cmd in &self.commands {
            if t >= cmd.t_start && t < cmd.t_end {
                let (x, y) = sample_pattern(&cmd.pattern, t);
                return (x, y, cmd.flow_rate);
            }
        }
        // No active command — drawdown / wait phase
        (0.0, 0.0, 0.0)
    }

    /// Total duration of the script (end time of last command).
    pub fn total_duration(&self) -> f64 {
        self.commands.iter().map(|c| c.t_end).fold(0.0, f64::max)
    }

    /// Total water volume dispensed by the script (mL).
    pub fn total_water_ml(&self) -> f64 {
        self.commands
            .iter()
            .map(|c| c.flow_rate * (c.t_end - c.t_start))
            .sum()
    }
}

/// Compute (x, y) in [-1, 1] for a given pattern at time t.
fn sample_pattern(pattern: &PourPattern, t: f64) -> (f64, f64) {
    match pattern {
        PourPattern::Center => (0.0, 0.0),

        PourPattern::Spiral {
            freq_hz,
            r_min,
            r_max,
        } => {
            if *freq_hz <= 0.0 {
                return (0.0, 0.0);
            }
            let angle = 2.0 * PI * freq_hz * t;
            // Radius oscillates between r_min and r_max using a triangle wave
            // so the spiral sweeps in and out smoothly.
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
            // Slow rotation around the ring, ~0.5 Hz
            let angle = 2.0 * PI * 0.5 * t;
            (radius * angle.cos(), radius * angle.sin())
        }

        PourPattern::Point { x, y } => (*x, *y),
    }
}

// ---------------------------------------------------------------------------
// Built-in recipes
// ---------------------------------------------------------------------------

/// Classic spiral pour: bloom center 0-10s, wait 10-40s, spiral 40-130s, drawdown.
///
/// Total water: ~300 mL (10s * 5 mL/s bloom + 90s * ~3.0 mL/s main).
pub fn classic_spiral() -> PourScript {
    PourScript {
        commands: vec![
            // Bloom: center pour, gentle flow
            PourCommand {
                t_start: 0.0,
                t_end: 10.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            // Wait for bloom (no pour, 10-40s) — handled by absence of command
            // Main pour: spiral pattern
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
            // Drawdown after 130s — no command, sample returns 0
        ],
    }
}

/// Center-only pour: bloom center 0-10s, wait 10-40s, center 40-130s, drawdown.
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

/// Pulse pour: bloom 0-8s, then 4 pulses of 15s each separated by 10s waits.
///
/// Timeline: bloom 0-8, wait 8-25, pulse1 25-40, wait 40-50,
///           pulse2 50-65, wait 65-75, pulse3 75-90, wait 90-100,
///           pulse4 100-115, drawdown.
pub fn pulse_pour() -> PourScript {
    PourScript {
        commands: vec![
            // Bloom
            PourCommand {
                t_start: 0.0,
                t_end: 8.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            // Pulse 1
            PourCommand {
                t_start: 25.0,
                t_end: 40.0,
                flow_rate: 5.0,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.3,
                    r_min: 0.1,
                    r_max: 0.6,
                },
            },
            // Pulse 2
            PourCommand {
                t_start: 50.0,
                t_end: 65.0,
                flow_rate: 5.0,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.3,
                    r_min: 0.1,
                    r_max: 0.6,
                },
            },
            // Pulse 3
            PourCommand {
                t_start: 75.0,
                t_end: 90.0,
                flow_rate: 5.0,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.3,
                    r_min: 0.1,
                    r_max: 0.6,
                },
            },
            // Pulse 4
            PourCommand {
                t_start: 100.0,
                t_end: 115.0,
                flow_rate: 5.0,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.3,
                    r_min: 0.1,
                    r_max: 0.6,
                },
            },
        ],
    }
}

/// Edge-heavy pour: bloom center 0-10s, wait 10-40s, ring pour at r=0.85 from 40-130s.
pub fn edge_heavy() -> PourScript {
    PourScript {
        commands: vec![
            // Bloom: center
            PourCommand {
                t_start: 0.0,
                t_end: 10.0,
                flow_rate: 5.0,
                pattern: PourPattern::Center,
            },
            // Main pour: ring at r = 0.85
            PourCommand {
                t_start: 40.0,
                t_end: 130.0,
                flow_rate: 3.0,
                pattern: PourPattern::Ring { radius: 0.85 },
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classic_spiral_bloom() {
        let script = classic_spiral();
        // During bloom (t=5), should be center pour with flow > 0
        let (x, y, flow) = script.sample(5.0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert!(flow > 0.0, "Should be pouring during bloom");
    }

    #[test]
    fn test_classic_spiral_wait() {
        let script = classic_spiral();
        // During wait (t=25), no pour
        let (_x, _y, flow) = script.sample(25.0);
        assert_eq!(flow, 0.0, "No pour during bloom wait");
    }

    #[test]
    fn test_classic_spiral_main_pour() {
        let script = classic_spiral();
        // During main pour (t=80), spiral pattern with flow > 0
        let (x, y, flow) = script.sample(80.0);
        assert!(flow > 0.0, "Should be pouring during main phase");
        // Spiral should produce non-zero radius at most times
        let r = (x * x + y * y).sqrt();
        // r could be small near center crossing, so just check it's bounded
        assert!(r <= 1.0, "Pour position should be within normalized bounds");
    }

    #[test]
    fn test_drawdown_returns_zero() {
        let script = classic_spiral();
        let (_x, _y, flow) = script.sample(200.0);
        assert_eq!(flow, 0.0, "No pour during drawdown");
    }

    #[test]
    fn test_center_only() {
        let script = center_only();
        let (x, y, flow) = script.sample(80.0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert!(flow > 0.0);
    }

    #[test]
    fn test_pulse_pour_gaps() {
        let script = pulse_pour();
        // Between pulses (t=45), no pour
        let (_, _, flow) = script.sample(45.0);
        assert_eq!(flow, 0.0, "No pour between pulses");

        // During pulse 2 (t=55), should be pouring
        let (_, _, flow) = script.sample(55.0);
        assert!(flow > 0.0, "Should be pouring during pulse 2");
    }

    #[test]
    fn test_edge_heavy_ring() {
        let script = edge_heavy();
        let (x, y, flow) = script.sample(80.0);
        assert!(flow > 0.0);
        let r = (x * x + y * y).sqrt();
        assert!(
            (r - 0.85).abs() < 0.01,
            "Ring pour should be at r=0.85, got r={r}"
        );
    }

    #[test]
    fn test_total_duration() {
        let script = classic_spiral();
        assert!((script.total_duration() - 130.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_water() {
        let script = classic_spiral();
        let water = script.total_water_ml();
        // bloom: 10s * 5 mL/s = 50 mL, main: 90s * 3 mL/s = 270 mL, total = 320 mL
        assert!(
            (water - 320.0).abs() < 1e-10,
            "Total water should be ~320 mL, got {water}"
        );
    }
}
