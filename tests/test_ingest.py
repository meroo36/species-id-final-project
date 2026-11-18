"""Ingestion keeps only openly licensed photos from the open-data bucket."""

from species_id import config, ingest

OPEN = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/square.jpg"
STATIC = "https://static.inaturalist.org/photos/{}/square.jpg"


def observation(obs_id: int, photo_id: int, license_code: str | None, url: str = OPEN) -> dict:
    return {
        "id": obs_id,
        "observed_on": "2026-01-01",
        "photos": [
            {
                "id": photo_id,
                "license_code": license_code,
                "url": url.format(photo_id),
                "attribution": "(c) someone, some rights reserved",
            }
        ],
    }


def fake_api(pages: list[list[dict]]):
    remaining = list(pages)

    def api_get(client, path, params):
        return {"results": remaining.pop(0) if remaining else []}

    return api_get


def test_filters_license_and_host(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_LICENSES", ["cc0", "cc-by"])
    page = [
        observation(10, 1, "cc-by"),
        observation(11, 2, None),                # all rights reserved
        observation(12, 3, "cc-by-nd"),          # license not allowed
        observation(13, 4, "cc-by", url=STATIC),  # not on the open-data bucket
        observation(14, 5, "cc0"),
    ]
    monkeypatch.setattr(ingest, "api_get", fake_api([page]))

    photos = list(ingest.candidate_photos(None, taxon_id=7, wanted=10, known=set()))

    assert [p["photo_id"] for p in photos] == [1, 5]
    assert all(p["attribution"] and p["license"] in {"cc0", "cc-by"} for p in photos)
    assert photos[0]["url"].endswith("/medium.jpg")


def test_skips_known_photos_and_stops_at_wanted(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_LICENSES", ["cc-by"])
    page = [observation(i, i, "cc-by") for i in range(1, 7)]
    monkeypatch.setattr(ingest, "api_get", fake_api([page]))

    photos = list(ingest.candidate_photos(None, taxon_id=7, wanted=2, known={1, 2}))

    assert [p["photo_id"] for p in photos] == [3, 4]


def test_split_is_deterministic(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_LICENSES", ["cc-by"])
    page = [observation(i, i, "cc-by") for i in (5, 6, 10)]
    monkeypatch.setattr(ingest, "api_get", fake_api([page]))

    splits = {p["observation_id"]: p["split"] for p in ingest.candidate_photos(None, 7, 10, set())}

    assert splits == {5: "test", 6: "train", 10: "test"}
