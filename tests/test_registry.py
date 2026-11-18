"""The production tag only moves when the new model is not worse than production."""

from types import SimpleNamespace

import pytest

from species_id import registry


class FakeClient:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.alias = None
        self.tags = {}

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics))

    def set_registered_model_alias(self, name, alias, version):
        self.alias = (alias, version)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[key] = value


@pytest.mark.parametrize(
    ("metrics", "promoted"),
    [
        ({"macro_f1": 0.90}, True),                               # first model ever
        ({"macro_f1": 0.92, "production_macro_f1": 0.90}, True),  # better
        ({"macro_f1": 0.90, "production_macro_f1": 0.90}, True),  # equal: prefer fresher data
        ({"macro_f1": 0.897, "production_macro_f1": 0.90}, True),  # within tolerance
        ({"macro_f1": 0.85, "production_macro_f1": 0.90}, False),  # clearly worse
    ],
)
def test_promote_if_better(monkeypatch, metrics, promoted):
    fake = FakeClient(metrics)
    monkeypatch.setattr(registry, "client", lambda: fake)

    decision = registry.promote_if_better("run-1", "7")

    assert decision["promoted"] is promoted
    assert fake.alias == (("production", "7") if promoted else None)
    assert fake.tags["validation_status"] == ("promoted" if promoted else "rejected")
