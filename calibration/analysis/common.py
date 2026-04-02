"""Shared physics helpers for calibration analysis.

Mirrors the Rust sim-core constants and provides a 1D PuckModel used by
both fit_darcy.py (flow prediction) and fit_extraction.py (kinetics fitting).
"""

import math

import numpy as np


# --- Physical constants (mirroring crates/sim-core/src/constants.rs) ---

WATER_DENSITY = 971.8       # kg/m³ at 93°C
GRAVITY = 9.81              # m/s²
VISC_A = 2.414e-5           # Pa·s
VISC_B = 247.8              # K
VISC_C = 140.0              # K

R_GAS = 8.314               # J/(mol·K)
T_REF = 366.15              # K (93°C)
T_AMBIENT = 295.0           # K (22°C)

C_SAT_REF = 250.0           # kg/m³ at T_REF
SOLUBLES_FAST_FRAC = 0.21
SOLUBLES_SLOW_FRAC = 0.09
K_FAST_REF = 5e-5           # m/s at T_REF
K_SLOW_REF = 2e-7           # m/s at T_REF
E_ACTIVATION = 65_000.0     # J/mol

COFFEE_DENSITY = 1100.0     # kg/m³ particle density
BASE_POROSITY = 0.40

# Espresso basket geometry
BASKET_DIAMETER = 0.058     # m (58mm)
BASKET_AREA = math.pi * (BASKET_DIAMETER / 2) ** 2  # m²


# --- Helper functions ---

def viscosity(t_k: float) -> float:
    """Dynamic viscosity of water (Pa·s) as function of temperature (K)."""
    return VISC_A * 10.0 ** (VISC_B / (t_k - VISC_C))


def arrhenius(k_ref: float, t_k: float, e_act: float = E_ACTIVATION) -> float:
    """Arrhenius rate scaling from reference temperature."""
    return k_ref * math.exp((e_act / R_GAS) * (1.0 / T_REF - 1.0 / t_k))


def c_saturation(t_k: float) -> float:
    """Saturation concentration (kg/m³) as function of temperature."""
    return C_SAT_REF * (1.0 + 0.01 * (t_k - T_REF))


def kozeny_carman(d_p: float, porosity: float) -> float:
    """Kozeny-Carman permeability: k = (ε³ · d_p²) / (180 · (1-ε)²)."""
    if porosity <= 0.0 or porosity >= 1.0 or d_p <= 0.0:
        return 0.0
    eps3 = porosity ** 3
    one_minus = 1.0 - porosity
    return eps3 * d_p ** 2 / (180.0 * one_minus ** 2)


def classify_phase_control(
    pressure_measured: np.ndarray,
    pressure_goal: np.ndarray | None,
    flow_measured: np.ndarray,
    flow_goal: np.ndarray | None,
) -> str:
    """Classify whether a phase is pressure-controlled or flow-controlled.

    Compares how closely measured values track their goals.
    Returns 'pressure_controlled', 'flow_controlled', or 'unknown'.
    """
    if len(pressure_measured) < 5:
        return "unknown"

    p_corr = -1.0
    f_corr = -1.0

    if pressure_goal is not None and len(pressure_goal) == len(pressure_measured):
        if np.std(pressure_measured) > 0.1 and np.std(pressure_goal) > 0.1:
            p_corr = float(np.corrcoef(pressure_measured, pressure_goal)[0, 1])

    if flow_goal is not None and len(flow_goal) == len(flow_measured):
        if np.std(flow_measured) > 0.01 and np.std(flow_goal) > 0.01:
            f_corr = float(np.corrcoef(flow_measured, flow_goal)[0, 1])

    if p_corr > f_corr and p_corr > 0.8:
        return "pressure_controlled"
    elif f_corr > p_corr and f_corr > 0.8:
        return "flow_controlled"
    return "unknown"


# --- 1D Puck Model ---

class PuckModel:
    """1D espresso puck model with N cells for Darcy flow and extraction.

    Used by both fit_darcy.py (flow prediction) and fit_extraction.py
    (coupled 1D extraction with spatial concentration gradients).
    """

    def __init__(self, n_cells: int = 20):
        self.n_cells = n_cells
        # Per-cell state (initialized by reset())
        self.k_cells = None
        self.porosity_cells = None
        self.fines_mobile = None
        self.fines_deposited = None
        # Extraction state (initialized by reset_extraction())
        self.m_fast = None
        self.m_slow = None
        self.concentration = None

    def reset(
        self,
        dose_g: float,
        d_p: float,
        porosity: float,
        k0_factor: float,
    ):
        """Initialize puck geometry and flow state.

        Args:
            dose_g: coffee dose in grams
            d_p: particle diameter in meters
            porosity: initial bed porosity
            k0_factor: multiplier on Kozeny-Carman base permeability
        """
        self.dose_kg = dose_g / 1000.0
        self.d_p = d_p
        self.porosity_base = porosity
        self.k0_factor = k0_factor

        # Puck thickness from dose and porosity
        self.L = self.dose_kg / (COFFEE_DENSITY * BASKET_AREA * (1.0 - porosity))
        self.dz = self.L / self.n_cells

        # Per-cell state
        k_base = kozeny_carman(d_p, porosity) * k0_factor
        self.k_cells = np.full(self.n_cells, k_base)
        self.porosity_cells = np.full(self.n_cells, porosity)

        # Fines: initial mobile fines concentrated in upper third
        self.fines_mobile = np.zeros(self.n_cells)
        self.fines_mobile[: self.n_cells // 3] = 0.02
        self.fines_deposited = np.zeros(self.n_cells)

    def reset_extraction(self, f_fast: float, f_slow: float):
        """Initialize extraction state for 1D coupled kinetics.

        Args:
            f_fast: fraction of coffee mass that is fast-extracting solubles
            f_slow: fraction of coffee mass that is slow-extracting solubles
        """
        mass_per_cell = self.dose_kg / self.n_cells
        self.m_fast = np.full(self.n_cells, f_fast * mass_per_cell)
        self.m_slow = np.full(self.n_cells, f_slow * mass_per_cell)
        self.m_fast_0 = self.m_fast.copy()
        self.m_slow_0 = self.m_slow.copy()
        # Concentration of dissolved solubles in liquid phase (kg/m³)
        self.concentration = np.zeros(self.n_cells)

    def compute_flow(self, pressure_pa: float, temp_k: float) -> float:
        """Compute flow rate (m³/s) given total pressure drop across puck.

        Uses Darcy's law with cells in series.
        """
        mu = viscosity(temp_k)
        if mu <= 0 or pressure_pa <= 0:
            return 0.0

        # Total resistance = sum of cell resistances in series
        # R_cell = dz / (k * A), R_total = sum(R_cell) * mu
        r_total = 0.0
        for j in range(self.n_cells):
            k = self.k_cells[j]
            if k <= 0:
                return 0.0  # clogged
            r_total += self.dz / (k * BASKET_AREA)
        r_total *= mu

        return pressure_pa / r_total if r_total > 0 else 0.0

    def compute_pressure(self, flow_m3s: float, temp_k: float) -> float:
        """Compute pressure (Pa) given flow rate through puck.

        Inverse of compute_flow: P = Q * R_total.
        """
        mu = viscosity(temp_k)
        r_total = 0.0
        for j in range(self.n_cells):
            k = self.k_cells[j]
            if k <= 0:
                return 1e7  # effectively infinite resistance
            r_total += self.dz / (k * BASKET_AREA)
        r_total *= mu
        return flow_m3s * r_total

    def update_fines(self, flow_m3s: float, dt: float, fines_rate: float, fines_capture: float):
        """Advect mobile fines downstream and update permeability.

        Args:
            flow_m3s: volumetric flow rate (m³/s)
            dt: timestep (s)
            fines_rate: rate of fines detachment (unused -- fines start pre-loaded)
            fines_capture: rate at which mobile fines re-lodge (1/s)
        """
        v_sup = flow_m3s / BASKET_AREA if flow_m3s > 0 else 0.0

        # Sweep bottom-up for upwind stability
        for j in range(self.n_cells - 1, 0, -1):
            transport = self.fines_mobile[j - 1] * v_sup * dt / self.dz
            capture = self.fines_mobile[j] * fines_capture * dt
            # Clamp transport to available fines
            transport = min(transport, self.fines_mobile[j - 1])
            capture = min(capture, self.fines_mobile[j])

            self.fines_mobile[j] += transport - capture
            self.fines_mobile[j - 1] -= transport
            self.fines_deposited[j] += capture

        # Also capture fines in cell 0 (they don't come from anywhere above)
        capture_0 = self.fines_mobile[0] * fines_capture * dt
        capture_0 = min(capture_0, self.fines_mobile[0])
        self.fines_mobile[0] -= capture_0
        self.fines_deposited[0] += capture_0

        # Update permeability from modified porosity
        for j in range(self.n_cells):
            eps_j = max(self.porosity_base - self.fines_deposited[j], 0.05)
            self.porosity_cells[j] = eps_j
            self.k_cells[j] = kozeny_carman(self.d_p, eps_j) * self.k0_factor

    def extraction_step(
        self,
        flow_m3s: float,
        temp_k: float,
        dt: float,
        k_fast: float,
        k_slow: float,
        e_act: float,
    ) -> float:
        """Run one extraction timestep with 1D spatial coupling.

        Each cell dissolves solubles locally, then flow carries dissolved
        concentration downstream (upwind advection).

        Returns total mass extracted this step (kg).
        """
        if self.m_fast is None:
            raise RuntimeError("Call reset_extraction() before extraction_step()")

        kf = arrhenius(k_fast, temp_k, e_act)
        ks = arrhenius(k_slow, temp_k, e_act)
        c_sat = c_saturation(temp_k)

        total_extracted = 0.0
        v_sup = flow_m3s / BASKET_AREA if flow_m3s > 0 else 0.0

        for j in range(self.n_cells):
            eps = self.porosity_cells[j]
            v_liquid = eps * BASKET_AREA * self.dz  # liquid volume in cell

            if v_liquid <= 0:
                continue

            c_local = self.concentration[j]
            driving = max(c_sat - c_local, 0.0)

            # Two-pool dissolution
            frac_fast = self.m_fast[j] / max(self.m_fast_0[j], 1e-30)
            frac_slow = self.m_slow[j] / max(self.m_slow_0[j], 1e-30)

            dm_fast = kf * driving * frac_fast * dt
            dm_slow = ks * driving * frac_slow * dt

            dm_fast = min(dm_fast, self.m_fast[j])
            dm_slow = min(dm_slow, self.m_slow[j])

            self.m_fast[j] -= dm_fast
            self.m_slow[j] -= dm_slow
            total_extracted += dm_fast + dm_slow

            # Add dissolved mass to local concentration
            self.concentration[j] += (dm_fast + dm_slow) / v_liquid

        # Upwind advection: flow carries concentration downstream (cell 0 -> cell N-1)
        if v_sup > 0 and dt > 0:
            # CFL-safe advection
            cfl_dt = 0.8 * self.dz / v_sup if v_sup > 0 else dt
            n_sub = max(1, int(math.ceil(dt / cfl_dt)))
            sub_dt = dt / n_sub

            for _ in range(n_sub):
                c_new = self.concentration.copy()
                for j in range(self.n_cells):
                    eps = self.porosity_cells[j]
                    # Flux in from upstream cell
                    c_up = self.concentration[j - 1] if j > 0 else 0.0
                    c_here = self.concentration[j]
                    # Net advective flux
                    flux = v_sup * (c_up - c_here) * sub_dt / self.dz
                    c_new[j] += flux
                    c_new[j] = max(c_new[j], 0.0)
                self.concentration[:] = c_new

        return total_extracted

    def simulate_shot(
        self,
        shot: dict,
        darcy_params: dict | None = None,
        extraction_params: dict | None = None,
    ) -> dict:
        """Simulate a full shot and return predicted time-series.

        Args:
            shot: normalized shot record from filter_and_store.py
            darcy_params: {k0_factor, porosity, fines_rate, fines_capture, d_p}
            extraction_params: {k_fast, k_slow, e_act, f_fast, f_slow}

        Returns dict with keys: predicted_flow_ml_s, predicted_pressure_bar,
            predicted_ey_pct, and per-timestep arrays.
        """
        dp = darcy_params or {}
        k0 = dp.get("k0_factor", 1.0)
        porosity = dp.get("porosity", BASE_POROSITY)
        fines_rate = dp.get("fines_rate", 0.01)
        fines_capture = dp.get("fines_capture", 0.1)
        d_p = dp.get("d_p", 300e-6)

        self.reset(shot["dose_g"], d_p, porosity, k0)

        do_extraction = extraction_params is not None
        if do_extraction:
            ep = extraction_params
            self.reset_extraction(ep["f_fast"], ep["f_slow"])

        time_s = np.array(shot["time_s"])
        pressure_bar = np.array(shot["pressure_bar"])
        flow_ml_s = np.array(shot["flow_ml_s"])
        temp_c = np.array(shot["temp_basket_c"])

        # Determine control mode per phase
        phases = shot.get("phases", [{"start_idx": 0, "end_idx": len(time_s) - 1, "state": "unknown"}])
        p_goal = np.array(shot["pressure_goal_bar"]) if shot.get("pressure_goal_bar") else None
        f_goal = np.array(shot["flow_goal_ml_s"]) if shot.get("flow_goal_ml_s") else None

        # Build per-timestep control mode array
        control = np.full(len(time_s), "unknown", dtype=object)
        for phase in phases:
            si, ei = phase["start_idx"], phase["end_idx"]
            if ei <= si:
                continue
            mode = classify_phase_control(
                pressure_bar[si:ei + 1],
                p_goal[si:ei + 1] if p_goal is not None else None,
                flow_ml_s[si:ei + 1],
                f_goal[si:ei + 1] if f_goal is not None else None,
            )
            control[si:ei + 1] = mode

        pred_flow = np.zeros(len(time_s))
        pred_pressure = np.zeros(len(time_s))
        total_extracted = 0.0

        for i in range(1, len(time_s)):
            dt = time_s[i] - time_s[i - 1]
            if dt <= 0:
                continue

            t_k = temp_c[i] + 273.15
            mode = control[i]

            if mode == "pressure_controlled":
                p_pa = pressure_bar[i] * 1e5
                q = self.compute_flow(p_pa, t_k)
                pred_flow[i] = q * 1e6  # m³/s -> mL/s
                pred_pressure[i] = pressure_bar[i]
            elif mode == "flow_controlled":
                q = flow_ml_s[i] * 1e-6  # mL/s -> m³/s
                p_pa = self.compute_pressure(q, t_k)
                pred_pressure[i] = p_pa / 1e5  # Pa -> bar
                pred_flow[i] = flow_ml_s[i]
                q = flow_ml_s[i] * 1e-6
            else:
                # Unknown: use measured flow for fines/extraction, predict nothing
                q = flow_ml_s[i] * 1e-6
                pred_flow[i] = flow_ml_s[i]
                pred_pressure[i] = pressure_bar[i]

            # Update fines migration
            self.update_fines(q, dt, fines_rate, fines_capture)

            # Extraction step
            if do_extraction:
                ep = extraction_params
                dm = self.extraction_step(q, t_k, dt, ep["k_fast"], ep["k_slow"], ep["e_act"])
                total_extracted += dm

        ey_pct = (total_extracted / self.dose_kg * 100.0) if self.dose_kg > 0 else 0.0

        return {
            "predicted_flow_ml_s": pred_flow.tolist(),
            "predicted_pressure_bar": pred_pressure.tolist(),
            "predicted_ey_pct": ey_pct,
            "control_modes": control.tolist(),
        }
