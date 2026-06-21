mod simulator;

#[cfg(target_arch = "wasm32")]
pub(crate) use simulator::Simulator;
