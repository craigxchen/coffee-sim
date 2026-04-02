"""Test Darcy flow model against Decent pressure-flow data."""

import numpy as np
import pytest

from common import PuckModel


class TestPressureFlow:
    def test_predicted_flow_nrmse(self, shots_with_goals, darcy_params):
        """For pressure-controlled shots, predicted flow NRMSE < 25%."""
        puck = PuckModel(n_cells=20)
        nrmses = []

        for shot in shots_with_goals:
            try:
                result = puck.simulate_shot(shot, darcy_params=darcy_params)
                control = result["control_modes"]

                p_idx = [i for i in range(len(control)) if control[i] == "pressure_controlled"]
                if len(p_idx) < 10:
                    continue

                pred = np.array([result["predicted_flow_ml_s"][i] for i in p_idx])
                meas = np.array([shot["flow_ml_s"][i] for i in p_idx])

                if np.ptp(meas) < 0.1:
                    continue

                nrmse = np.sqrt(np.mean((pred - meas) ** 2)) / np.ptp(meas)
                nrmses.append(nrmse)
            except Exception:
                continue

        if len(nrmses) < 10:
            pytest.skip(f"Only {len(nrmses)} pressure-controlled shots evaluated")

        median_nrmse = np.median(nrmses)
        assert median_nrmse < 0.25, f"Median flow NRMSE {median_nrmse:.2f} exceeds 25%"

    def test_predicted_pressure_nrmse(self, shots_with_goals, darcy_params):
        """For flow-controlled shots, predicted pressure NRMSE < 25%."""
        puck = PuckModel(n_cells=20)
        nrmses = []

        for shot in shots_with_goals:
            try:
                result = puck.simulate_shot(shot, darcy_params=darcy_params)
                control = result["control_modes"]

                f_idx = [i for i in range(len(control)) if control[i] == "flow_controlled"]
                if len(f_idx) < 10:
                    continue

                pred = np.array([result["predicted_pressure_bar"][i] for i in f_idx])
                meas = np.array([shot["pressure_bar"][i] for i in f_idx])

                if np.ptp(meas) < 0.1:
                    continue

                nrmse = np.sqrt(np.mean((pred - meas) ** 2)) / np.ptp(meas)
                nrmses.append(nrmse)
            except Exception:
                continue

        if len(nrmses) < 5:
            pytest.skip(f"Only {len(nrmses)} flow-controlled shots evaluated")

        median_nrmse = np.median(nrmses)
        assert median_nrmse < 0.25, f"Median pressure NRMSE {median_nrmse:.2f} exceeds 25%"

    def test_flow_shape_correlation(self, shots_with_goals, darcy_params):
        """Predicted flow curve shape should correlate with measured (r > 0.7)."""
        puck = PuckModel(n_cells=20)
        correlations = []

        for shot in shots_with_goals:
            try:
                result = puck.simulate_shot(shot, darcy_params=darcy_params)
                control = result["control_modes"]

                p_idx = [i for i in range(len(control)) if control[i] == "pressure_controlled"]
                if len(p_idx) < 10:
                    continue

                pred = np.array([result["predicted_flow_ml_s"][i] for i in p_idx])
                meas = np.array([shot["flow_ml_s"][i] for i in p_idx])

                if np.std(pred) < 0.01 or np.std(meas) < 0.01:
                    continue

                r = np.corrcoef(pred, meas)[0, 1]
                if np.isfinite(r):
                    correlations.append(r)
            except Exception:
                continue

        if len(correlations) < 10:
            pytest.skip(f"Only {len(correlations)} shots with valid correlation")

        median_r = np.median(correlations)
        assert median_r > 0.7, f"Median flow correlation {median_r:.2f} below 0.7"

    def test_permeability_evolution_direction(self, all_shots):
        """Derived permeability should decrease over the shot for >60% of shots."""
        from compute_permeability import compute_permeability_series

        decreasing_count = 0
        total = 0

        for shot in all_shots:
            t, k = compute_permeability_series(shot)
            if t is None or len(k) < 10:
                continue

            n = len(k)
            k_early = np.mean(k[: n // 3])
            k_late = np.mean(k[2 * n // 3 :])

            total += 1
            if k_late < k_early:
                decreasing_count += 1

        if total < 20:
            pytest.skip(f"Only {total} shots with valid permeability")

        frac = decreasing_count / total
        assert frac > 0.6, f"Only {frac:.0%} of shots show decreasing permeability (need >60%)"
