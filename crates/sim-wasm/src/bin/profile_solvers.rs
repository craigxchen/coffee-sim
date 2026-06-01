#[cfg(not(feature = "native-profiler"))]
compile_error!("build this binary with `--features native-profiler`");

#[cfg(feature = "native-profiler")]
fn main() {
    coffee_sim_wasm::mpm_3d::profiler::run_profile_from_env_args();
}
