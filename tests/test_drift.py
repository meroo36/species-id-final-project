"""Drift report on synthetic features, and the retraining decision."""

import numpy as np
import pytest

from species_id import config, drift

rng = np.random.default_rng(0)
REFERENCE = rng.normal(size=(1500, 64)).astype(np.float32)


def test_same_distribution_is_not_drift():
    current = rng.normal(size=(100, 64)).astype(np.float32)
    result, _ = drift.run_report(REFERENCE, current)
    assert result["dataset_drift"] is False


def test_shifted_distribution_is_drift():
    current = (rng.normal(size=(100, 64)) + 2.0).astype(np.float32)
    result, _ = drift.run_report(REFERENCE, current)
    assert result["dataset_drift"] is True
    assert result["share_of_drifted_columns"] > 0.5


@pytest.mark.parametrize(
    ("model", "drift_result", "force", "retrain"),
    [
        (None, {"drift_detected": False}, False, True),                          # first model
        ({"params": {}}, {"drift_detected": False}, False, True),                # unknown cutoff
        ({"params": {"data_cutoff": "x"}}, {"drift_detected": False}, False, False),
        ({"params": {"data_cutoff": "x"}}, {"drift_detected": True, "drift_share": 0.8}, False, True),
        ({"params": {"data_cutoff": "x"}}, {"drift_detected": False}, True, True),  # forced
    ],
)
def test_retrain_decision(monkeypatch, model, drift_result, force, retrain):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args):
            return self

        def fetchone(self):
            return (config.RETRAIN_MIN_NEW_PHOTOS - 1,)  # not enough new photos on their own

    monkeypatch.setattr(drift.registry, "production_info", lambda: model)
    monkeypatch.setattr(drift.db, "connect", lambda: FakeConnection())

    decision = drift.retrain_decision(drift_result, force=force)

    assert decision["retrain"] is retrain
    assert bool(decision["reasons"]) is retrain
