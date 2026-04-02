"""Shared fixtures for calibration tests.

Loads curated datasets and gracefully skips tests when the dataset
is too small for meaningful statistics.
"""

import json
import sys
from pathlib import Path

import pytest

# Add analysis directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

CURATED_DIR = Path(__file__).parent.parent / "curated"


def _load_json(filename: str) -> list[dict]:
    path = CURATED_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_fit_result(filename: str) -> dict | None:
    path = CURATED_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def all_shots() -> list[dict]:
    shots = _load_json("all_shots.json")
    if not shots:
        pytest.skip("No curated dataset found. Run filter_and_store.py first.")
    return shots


@pytest.fixture(scope="session")
def shots_with_ey() -> list[dict]:
    shots = _load_json("shots_with_ey.json")
    shots = [s for s in shots if s.get("ey_pct") and s["ey_pct"] > 0]
    if len(shots) < 10:
        pytest.skip(f"Only {len(shots)} shots with EY data (need >= 10)")
    return shots


@pytest.fixture(scope="session")
def shots_with_goals(all_shots) -> list[dict]:
    shots = [s for s in all_shots if s.get("pressure_goal_bar") or s.get("flow_goal_ml_s")]
    if len(shots) < 30:
        pytest.skip(f"Only {len(shots)} shots with goal data (need >= 30)")
    return shots


@pytest.fixture(scope="session")
def darcy_params() -> dict:
    result = _load_fit_result("darcy_fit_result.json")
    if result is None:
        pytest.skip("Darcy fit not yet run. Run fit_darcy.py first.")
    return {
        "k0_factor": result["k0_factor"],
        "porosity": result["porosity"],
        "fines_rate": result["fines_rate"],
        "fines_capture": result["fines_capture"],
        "d_p": result["d_p_m"],
    }


@pytest.fixture(scope="session")
def extraction_params() -> dict:
    result = _load_fit_result("extraction_fit_result.json")
    if result is None:
        pytest.skip("Extraction fit not yet run. Run fit_extraction.py first.")
    return result


@pytest.fixture(scope="session")
def thermal_params() -> dict:
    result = _load_fit_result("thermal_fit_result.json")
    if result is None:
        pytest.skip("Thermal fit not yet run. Run fit_thermal.py first.")
    return result
