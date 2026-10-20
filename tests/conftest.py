"""Shared pytest fixtures for AdaptiveAI suites.

Step 58: the project has a legitimate live deployment (production
pointer, registry versions, development database with live prediction
and promotion history). Every test is guarded automatically against
modifying the production models directory and the MLflow registry.
The development database file is deliberately NOT file-guarded: live
predictions and monitoring checks write to it by design, so concurrent
live use during a run would false-trip the guard. Database isolation
instead rests on construction (no test builds repositories on the
default development path; all tests use temporary databases).
"""

from pathlib import Path

import pytest

from app.core.config import get_project_root

_BASELINE = {}


def _live_state_snapshot():
    """Snapshots live repository file state (no content read).

    Covers the production models directory listing and the
    registered-model version set, so the live deployment (production
    pointer, HousePriceModel versions) is never modified by tests.
    """
    root = Path(get_project_root())
    models_dir = root / "models" / "house_price"
    models_state = (
        sorted(p.name for p in models_dir.iterdir()) if models_dir.exists() else []
    )
    registry_dir = root / "mlruns" / "models" / "HousePriceModel"
    registry_state = (
        sorted(p.name for p in registry_dir.iterdir()) if registry_dir.exists() else []
    )
    return {
        "models": models_state,
        "registry": registry_state,
    }


@pytest.fixture(scope="session", autouse=True)
def _record_live_baseline():
    """Records live repository state before any test runs in the session."""
    _BASELINE.update(_live_state_snapshot())
    yield
    assert _live_state_snapshot() == _BASELINE, (
        "Live repository state was modified during the test session; "
        "tests must use temporary model directories and MLflow stores only."
    )


@pytest.fixture(autouse=True)
def _guard_live_state_unchanged(_record_live_baseline):
    """Fails any test that modifies live repository state."""
    yield
    assert _live_state_snapshot() == _BASELINE, (
        "Live repository state was modified during the test session; "
        "tests must use temporary model directories and MLflow stores only."
    )
