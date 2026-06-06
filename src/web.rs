#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = initCoffeeSim)]
pub fn init_coffee_sim() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

#[wasm_bindgen(js_name = coffeeSimWasmReady)]
pub fn coffee_sim_wasm_ready() -> bool {
    true
}
