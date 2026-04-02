"""Test that Kozeny-Carman permeability is within order of magnitude of Decent-derived values."""

import numpy as np
import pytest

from common import kozeny_carman
from compute_permeability import compute_permeability_series


class TestPermeabilityRange:
    def test_kozeny_carman_within_order_of_magnitude(self, all_shots, darcy_params):
        """K-C predicted permeability should be within 10x of Decent-derived values."""
        d_p = darcy_params["d_p"]
        porosity = darcy_params["porosity"]
        k0_factor = darcy_params["k0_factor"]

        k_kc = kozeny_carman(d_p, porosity) * k0_factor

        ratios = []
        for shot in all_shots[:100]:  # spot check
            _, k = compute_permeability_series(shot)
            if k is None or len(k) < 5:
                continue

            k_median = np.median(k)
            if k_median <= 0:
                continue

            ratio = k_median / k_kc
            ratios.append(ratio)

        if len(ratios) < 10:
            pytest.skip(f"Only {len(ratios)} shots with valid permeability")

        median_ratio = np.median(ratios)
        # With the fitted k0_factor correction, the ratio should be close to 1
        # but real pucks vary, so allow a factor of 10
        assert 0.1 < median_ratio < 10.0, (
            f"Median K-C ratio {median_ratio:.2f} outside expected range [0.1, 10]"
        )

    def test_permeability_physical_range(self, all_shots):
        """Derived permeability values should be in physically reasonable range.

        Espresso puck permeability: ~1e-14 to 1e-11 m²
        """
        k_values = []
        for shot in all_shots[:100]:
            _, k = compute_permeability_series(shot)
            if k is None:
                continue
            k_values.extend(k.tolist())

        if len(k_values) < 50:
            pytest.skip(f"Only {len(k_values)} permeability data points")

        k_arr = np.array(k_values)
        k_arr = k_arr[k_arr > 0]  # filter zeros

        median_k = np.median(k_arr)
        assert 1e-15 < median_k < 1e-10, (
            f"Median permeability {median_k:.2e} m² outside espresso range [1e-15, 1e-10]"
        )
