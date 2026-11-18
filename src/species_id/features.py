"""Frozen pre-trained feature extractor (timm)."""

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import timm
import torch
from PIL import Image

BATCH_SIZE = 32


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=2)
def _load(extractor: str):
    model = timm.create_model(extractor, pretrained=True, num_classes=0).eval().to(_device())
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    return model, transform


def embed(images: Iterable[Image.Image], extractor: str) -> np.ndarray:
    """Return one feature vector per image, shape (n_images, n_features)."""
    model, transform = _load(extractor)
    device = next(model.parameters()).device
    images = list(images)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), BATCH_SIZE):
            batch = torch.stack(
                [transform(img.convert("RGB")) for img in images[start : start + BATCH_SIZE]]
            )
            out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
