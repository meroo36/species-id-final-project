"""Ingest openly licensed, research-grade photos from iNaturalist into the database.

Run with:  uv run python -m species_id.ingest

Safe to re-run: photos already in the database are skipped, so it tops each taxon up to
IMAGES_PER_TAXON. Only photos with an allowed license that are hosted on the open-data
bucket are kept, and their attribution is stored alongside them.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import httpx

from species_id import config, db

log = logging.getLogger("ingest")

API = "https://api.inaturalist.org/v1"
OPEN_DATA_HOST = "inaturalist-open-data.s3.amazonaws.com"
HEADERS = {"User-Agent": "species-id-mlops-student-project"}
API_PAUSE_S = 1.1  # the API allows 60 requests per minute
API_RETRIES = 5
TEST_EVERY = 5    # one observation in five goes to the test split


def api_get(client: httpx.Client, path: str, params: dict) -> dict:
    for attempt in range(API_RETRIES):
        try:
            response = client.get(f"{API}{path}", params=params)
            response.raise_for_status()
            time.sleep(API_PAUSE_S)
            return response.json()
        except httpx.HTTPError as error:
            if attempt == API_RETRIES - 1:
                raise
            wait = 5 * 2**attempt
            log.warning("API call failed (%s), retrying in %ds", error, wait)
            time.sleep(wait)


def resolve_taxon(client: httpx.Client, name: str) -> dict:
    results = api_get(client, "/taxa", {"q": name, "rank": "species", "per_page": 10})["results"]
    for taxon in results:
        if taxon["name"].lower() == name.lower():
            return {
                "taxon_id": taxon["id"],
                "scientific_name": taxon["name"],
                "common_name": taxon.get("preferred_common_name"),
            }
    raise ValueError(f"Taxon not found on iNaturalist: {name}")


def candidate_photos(client: httpx.Client, taxon_id: int, wanted: int, known: set[int]):
    """Yield up to `wanted` new photos (one per observation), newest observations first."""
    found, id_below = 0, None
    while found < wanted:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photo_license": ",".join(config.ALLOWED_LICENSES),
            "photos": "true",
            "per_page": 200,
            "order_by": "id",
            "order": "desc",
        }
        if id_below:
            params["id_below"] = id_below
        observations = api_get(client, "/observations", params)["results"]
        if not observations:
            return
        id_below = observations[-1]["id"]
        for obs in observations:
            photo = next(
                (
                    p
                    for p in obs["photos"]
                    if p.get("license_code") in config.ALLOWED_LICENSES
                    and urlparse(p["url"]).hostname == OPEN_DATA_HOST
                ),
                None,
            )
            if photo is None or photo["id"] in known:
                continue
            known.add(photo["id"])
            found += 1
            yield {
                "photo_id": photo["id"],
                "observation_id": obs["id"],
                "taxon_id": taxon_id,
                "url": photo["url"].replace("/square.", "/medium."),
                "license": photo["license_code"],
                "attribution": photo["attribution"],
                "observed_on": obs.get("observed_on"),
                "split": "test" if obs["id"] % TEST_EVERY == 0 else "train",
            }
            if found >= wanted:
                return


def download(client: httpx.Client, photo: dict) -> dict | None:
    target = config.IMAGE_DIR / str(photo["taxon_id"]) / f"{photo['photo_id']}.jpg"
    try:
        response = client.get(photo["url"])
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    except httpx.HTTPError as error:
        log.warning("Skipping photo %s: %s", photo["photo_id"], error)
        return None
    return {**photo, "local_path": str(target.relative_to(config.DATA_DIR))}


def ingest(
    images_per_taxon: int = config.IMAGES_PER_TAXON, new_per_taxon: int | None = None
) -> dict[str, int]:
    """Load photos for every configured taxon. Returns the number of new photos per taxon.

    By default each taxon is topped up to `images_per_taxon` photos (the one-time load).
    With `new_per_taxon`, that many additional photos are fetched instead, newest first
    (the scheduled pipeline uses this to keep the dataset growing).
    """
    db.init_schema()
    added: dict[str, int] = {}
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client, db.connect() as conn:
        for name in config.TAXA:
            taxon = resolve_taxon(client, name)
            conn.execute(
                """INSERT INTO taxa (taxon_id, scientific_name, common_name)
                   VALUES (%(taxon_id)s, %(scientific_name)s, %(common_name)s)
                   ON CONFLICT (taxon_id) DO UPDATE SET common_name = EXCLUDED.common_name""",
                taxon,
            )
            known = {
                row[0]
                for row in conn.execute(
                    "SELECT photo_id FROM photos WHERE taxon_id = %s", (taxon["taxon_id"],)
                )
            }
            wanted = new_per_taxon if new_per_taxon is not None else images_per_taxon - len(known)
            photos = list(candidate_photos(client, taxon["taxon_id"], wanted, known)) if wanted > 0 else []
            with ThreadPoolExecutor(max_workers=8) as pool:
                downloaded = [p for p in pool.map(lambda p: download(client, p), photos) if p]
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO photos (photo_id, observation_id, taxon_id, url, license,
                                           attribution, observed_on, local_path, split)
                       VALUES (%(photo_id)s, %(observation_id)s, %(taxon_id)s, %(url)s, %(license)s,
                               %(attribution)s, %(observed_on)s, %(local_path)s, %(split)s)
                       ON CONFLICT (photo_id) DO NOTHING""",
                    downloaded,
                )
            conn.commit()
            added[name] = len(downloaded)
            log.info("%s: %d new photos", name, len(downloaded))
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ingest()
