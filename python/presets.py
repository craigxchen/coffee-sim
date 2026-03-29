"""Preset configurations for common brewing methods."""

from .config import GrindProfile, PouroverParams, EspressoParams


def v60_default() -> PouroverParams:
    """Hario V60, 20g coffee, 320g water, medium grind."""
    return PouroverParams(
        coffee_mass_g=20.0,
        water_mass_g=320.0,
        water_temp_c=93.0,
        bloom_water_g=50.0,
        bloom_time_s=35.0,
        pour_rate_ml_s=4.0,
        pour_pattern="spiral",
        geometry="v60",
        grind=GrindProfile(d_main_um=700, d_fines_um=50, fines_fraction=0.15),
    )


def kalita_default() -> PouroverParams:
    """Kalita Wave, 20g coffee, 320g water, medium grind."""
    return PouroverParams(
        coffee_mass_g=20.0,
        water_mass_g=320.0,
        water_temp_c=92.0,
        bloom_water_g=50.0,
        bloom_time_s=35.0,
        pour_rate_ml_s=3.5,
        pour_pattern="center",
        geometry="kalita",
        grind=GrindProfile(d_main_um=750, d_fines_um=50, fines_fraction=0.12),
    )


def espresso_default() -> EspressoParams:
    """Standard 9-bar espresso, 18g in, 36g out."""
    return EspressoParams(
        coffee_mass_g=18.0,
        target_yield_g=36.0,
        water_temp_c=93.0,
        pressure_bar=9.0,
        preinfusion_bar=2.0,
        preinfusion_time_s=5.0,
        pressure_profile="flat",
        basket_diameter_mm=58.0,
        grind=GrindProfile(d_main_um=300, d_fines_um=30, fines_fraction=0.20),
    )
