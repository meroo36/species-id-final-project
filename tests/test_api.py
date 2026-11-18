"""API behaviour that does not need a database or a trained model."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from species_id import api, predict

client = TestClient(api.app)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_rejects_non_image():
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image")})
    assert response.status_code == 400


def test_predict_without_production_model(monkeypatch):
    def no_model(image):
        raise LookupError("No production model yet.")

    monkeypatch.setattr(predict, "predict", no_model)
    response = client.post("/predict", files={"file": ("leaf.png", png_bytes())})
    assert response.status_code == 503


@pytest.mark.parametrize(("confidence", "needs_review"), [(0.95, False), (0.30, True)])
def test_predict_returns_review_flag(monkeypatch, confidence, needs_review):
    result = {"prediction": {"taxon_id": 1}, "confidence": confidence, "needs_review": needs_review}
    monkeypatch.setattr(predict, "predict", lambda image: result)
    response = client.post("/predict", files={"file": ("leaf.png", png_bytes())})
    assert response.status_code == 200
    assert response.json()["needs_review"] is needs_review
