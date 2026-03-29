"""Brew parameter dataclasses."""

from dataclasses import dataclass, field


@dataclass
class GrindProfile:
    d_main_um: float = 700.0          # median main grind size (μm)
    d_fines_um: float = 50.0          # median fines size (μm)
    fines_fraction: float = 0.15      # mass fraction of fines
    sigma_main: float = 0.2           # log-normal spread
    sigma_fines: float = 0.3

    def to_dict(self) -> dict:
        return {
            "d_main_um": self.d_main_um,
            "d_fines_um": self.d_fines_um,
            "fines_fraction": self.fines_fraction,
            "sigma_main": self.sigma_main,
            "sigma_fines": self.sigma_fines,
        }


@dataclass
class PouroverParams:
    coffee_mass_g: float = 20.0
    water_mass_g: float = 320.0       # 1:16 ratio
    water_temp_c: float = 93.0
    bloom_water_g: float = 50.0
    bloom_time_s: float = 35.0
    pour_rate_ml_s: float = 4.0       # main pour rate
    pour_pattern: str = "spiral"      # "center", "spiral", "pulse"
    geometry: str = "v60"
    grind: GrindProfile = field(default_factory=GrindProfile)

    # Grid resolution
    grid_nx: int = 40
    grid_ny: int = 40
    grid_nz: int = 30
    grid_dx: float = 0.001            # 1mm voxel size


@dataclass
class EspressoParams:
    coffee_mass_g: float = 18.0
    target_yield_g: float = 36.0      # 1:2 ratio
    water_temp_c: float = 93.0
    pressure_bar: float = 9.0
    preinfusion_bar: float = 2.0
    preinfusion_time_s: float = 5.0
    pressure_profile: str = "flat"    # "flat", "declining", "ramp"
    basket_diameter_mm: float = 58.0
    compressibility_alpha: float = 1e-7  # Pa⁻¹, bed compressibility coefficient
    grind: GrindProfile = field(
        default_factory=lambda: GrindProfile(d_main_um=300, fines_fraction=0.20)
    )

    # Grid resolution
    grid_nx: int = 60
    grid_ny: int = 60
    grid_nz: int = 40
    grid_dx: float = 0.0005           # 0.5mm voxel size
