use std::fmt;
use std::str::FromStr;

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SolverSpec {
    Mpm,
}

impl SolverSpec {
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Mpm => "mpm",
        }
    }

    pub(crate) fn runnable_specs() -> &'static [Self] {
        &[Self::Mpm]
    }
}

impl fmt::Display for SolverSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for SolverSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "mpm" | "mpm:rbgs" | "rbgs" => Ok(Self::Mpm),
            other => Err(format!(
                "unknown profiler solver '{other}'; available solvers: mpm"
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SolverRunMetadata {
    pub(crate) requested: Vec<String>,
    pub(crate) current: String,
    pub(crate) ordinal: usize,
    pub(crate) count: usize,
    pub(crate) multiple_outputs: bool,
}

pub(crate) fn parse_solver_specs(value: &str) -> Result<Vec<SolverSpec>, String> {
    if value.trim().eq_ignore_ascii_case("all") {
        return Ok(SolverSpec::runnable_specs().to_vec());
    }
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect()
}

pub(crate) fn solver_list_summary() -> String {
    let mut summary = String::from("available profiler solvers:\n");
    for solver in SolverSpec::runnable_specs() {
        summary.push_str("  ");
        summary.push_str(solver.id());
        summary.push('\n');
    }
    summary.push_str("  all\n");
    summary
}

pub(crate) fn solver_run_metadata(
    solvers: &[SolverSpec],
    current: SolverSpec,
) -> SolverRunMetadata {
    let requested = solvers
        .iter()
        .map(|solver| solver.id().to_string())
        .collect::<Vec<_>>();
    let current_id = current.id().to_string();
    let ordinal = requested
        .iter()
        .position(|id| id == &current_id)
        .map(|index| index + 1)
        .unwrap_or(1);
    SolverRunMetadata {
        requested,
        current: current_id,
        ordinal,
        count: solvers.len(),
        multiple_outputs: solvers.len() > 1,
    }
}

pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_solver_parser_accepts_mpm_and_all() {
        assert_eq!(parse_solver_specs("mpm"), Ok(vec![SolverSpec::Mpm]));
        assert_eq!(parse_solver_specs("mpm:rbgs"), Ok(vec![SolverSpec::Mpm]));
        assert_eq!(parse_solver_specs("all"), Ok(vec![SolverSpec::Mpm]));
    }

    #[test]
    fn profiler_solver_parser_rejects_unregistered_solvers() {
        assert!("xpbd".parse::<SolverSpec>().is_err());
        assert!(parse_solver_specs("dfsph").is_err());
    }

    #[test]
    fn solver_run_metadata_tracks_single_mpm_run() {
        let solvers = parse_solver_specs("all").unwrap();
        let metadata = solver_run_metadata(&solvers, SolverSpec::Mpm);
        assert_eq!(metadata.requested, vec!["mpm"]);
        assert_eq!(metadata.current, "mpm");
        assert_eq!(metadata.ordinal, 1);
        assert_eq!(metadata.count, 1);
        assert!(!metadata.multiple_outputs);
    }
}
