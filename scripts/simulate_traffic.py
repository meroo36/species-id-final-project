"""Send photos from the local dataset to the API, to exercise monitoring.

    uv run python scripts/simulate_traffic.py --n 100            # normal photos
    uv run python scripts/simulate_traffic.py --n 100 --drift    # degraded photos

With --drift, photos are turned grey, blurred and darkened before they are sent. This shifts
their features, lowers confidence, and should trip both the Evidently prediction-drift check
and the Grafana human-review alert.
"""

import argparse
import io
import random
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def degrade(image: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(image).convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=6))
    return ImageEnhance.Brightness(image).enhance(0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50, help="number of photos to send")
    parser.add_argument("--drift", action="store_true", help="degrade photos before sending")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", default="data/images")
    args = parser.parse_args()

    photos = random.sample(sorted(Path(args.data).glob("*/*.jpg")), args.n)
    review = 0
    with httpx.Client(timeout=60) as client:
        for path in photos:
            image = Image.open(path).convert("RGB")
            if args.drift:
                image = degrade(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            response = client.post(
                f"{args.api}/predict", files={"file": (path.name, buffer.getvalue())}
            )
            response.raise_for_status()
            review += response.json()["needs_review"]
    print(f"Sent {len(photos)} photos; {review} flagged for human review ({review / len(photos):.0%}).")


if __name__ == "__main__":
    main()
