"""Dataset versioning with DVC (no Git).

The DVC project lives inside DATA_DIR. Each snapshot hashes the image directory; the hash
identifies the exact set of photos a model was trained on and is logged to MLflow.
"""

import subprocess
import sys

import yaml

from species_id import config


def _dvc(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        cwd=config.DATA_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def snapshot_dataset() -> dict:
    """Record the current state of the image directory and return its DVC metadata."""
    if not (config.DATA_DIR / ".dvc").exists():
        _dvc("init", "--no-scm", "-q")
        _dvc("config", "core.analytics", "false")
        # Link instead of copy where the filesystem allows it, so the cache costs no extra disk.
        _dvc("config", "cache.type", "reflink,hardlink,copy")
    _dvc("add", "-q", config.IMAGE_DIR.name)
    dvc_file = config.DATA_DIR / f"{config.IMAGE_DIR.name}.dvc"
    out = yaml.safe_load(dvc_file.read_text())["outs"][0]
    return {
        "md5": out["md5"],
        "nfiles": out["nfiles"],
        "size_bytes": out["size"],
        "dvc_file": dvc_file,
    }
