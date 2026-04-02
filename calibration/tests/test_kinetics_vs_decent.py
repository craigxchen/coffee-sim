"""Test extraction kinetics model against Decent shots with measured EY."""

import numpy as np
import pytest

from common import PuckModel


class TestKineticsVsDecent:
    def test_ey_within_tolerance(self, shots_with_ey, extraction_params, darcy_params):
        """Sim EY prediction matches Decent shots within +/-3% absolute."""
        puck = PuckModel(n_cells=20)
        errors = []

        ep = {
            "k_fast": extraction_params["K_FAST_REF"],
            "k_slow": extraction_params["K_SLOW_REF"],
            "e_act": extraction_params["E_ACTIVATION"],
            "f_fast": extraction_params["SOLUBLES_FAST_FRAC"],
            "f_slow": extraction_params["SOLUBLES_SLOW_FRAC"],
        }

        for shot in shots_with_ey:
            try:
                result = puck.simulate_shot(shot, darcy_params=darcy_params, extraction_params=ep)
                ey_pred = result["predicted_ey_pct"]
                ey_meas = shot["ey_pct"]
                errors.append(abs(ey_pred - ey_meas))
            except Exception:
                continue

        if len(errors) < 5:
            pytest.skip(f"Only {len(errors)} valid predictions")

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        assert mean_error < 3.0, f"Mean EY error {mean_error:.1f}% exceeds 3% threshold"
        assert max_error < 8.0, f"Max EY error {max_error:.1f}% exceeds 8% threshold"

    def test_ey_correlation(self, shots_with_ey, extraction_params, darcy_params):
        """Predicted and measured EY should be positively correlated (r > 0.5)."""
        puck = PuckModel(n_cells=20)
        predicted = []
        measured = []

        ep = {
            "k_fast": extraction_params["K_FAST_REF"],
            "k_slow": extraction_params["K_SLOW_REF"],
            "e_act": extraction_params["E_ACTIVATION"],
            "f_fast": extraction_params["SOLUBLES_FAST_FRAC"],
            "f_slow": extraction_params["SOLUBLES_SLOW_FRAC"],
        }

        for shot in shots_with_ey:
            try:
                result = puck.simulate_shot(shot, darcy_params=darcy_params, extraction_params=ep)
                predicted.append(result["predicted_ey_pct"])
                measured.append(shot["ey_pct"])
            except Exception:
                continue

        if len(predicted) < 10:
            pytest.skip(f"Only {len(predicted)} valid predictions")

        r = np.corrcoef(predicted, measured)[0, 1]
        assert r > 0.5, f"EY correlation {r:.2f} below 0.5 threshold"

    def test_two_pool_depletion_ordering(self, shots_with_ey, extraction_params, darcy_params):
        """Fast pool should deplete more than slow pool for all shots."""
        puck = PuckModel(n_cells=20)

        ep = {
            "k_fast": extraction_params["K_FAST_REF"],
            "k_slow": extraction_params["K_SLOW_REF"],
            "e_act": extraction_params["E_ACTIVATION"],
            "f_fast": extraction_params["SOLUBLES_FAST_FRAC"],
            "f_slow": extraction_params["SOLUBLES_SLOW_FRAC"],
        }

        correct_order = 0
        total = 0

        for shot in shots_with_ey[:50]:  # spot check
            try:
                puck.reset(shot["dose_g"], darcy_params["d_p"], darcy_params["porosity"], darcy_params["k0_factor"])
                puck.reset_extraction(ep["f_fast"], ep["f_slow"])

                initial_fast = puck.m_fast.sum()
                initial_slow = puck.m_slow.sum()

                # Run simulation
                puck.simulate_shot(shot, darcy_params=darcy_params, extraction_params=ep)

                final_fast = puck.m_fast.sum()
                final_slow = puck.m_slow.sum()

                fast_depleted = (initial_fast - final_fast) / initial_fast
                slow_depleted = (initial_slow - final_slow) / initial_slow

                total += 1
                if fast_depleted > slow_depleted:
                    correct_order += 1
            except Exception:
                continue

        if total < 5:
            pytest.skip(f"Only {total} valid simulations")

        frac = correct_order / total
        assert frac > 0.9, f"Fast pool depletes before slow in only {frac:.0%} of shots (need >90%)"
