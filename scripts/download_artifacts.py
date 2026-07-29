"""Download and verify the EOG/EMG subset of EEGdenoiseNet.

Dataset contents are written under ignored ``data/`` and are never committed.
The pinned Git commit and per-file SHA-256/size/shape are recorded in a manifest.
EEGdenoiseNet's GitHub arrays are 2-second, 256 Hz single-channel epochs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np


COMMIT = "8d290661146c7189c98cc04812d37371d4b9426c"
FILES = {
    "EOG_all_epochs.npy": (3400, 512),
    "EMG_all_epochs.npy": (5598, 512),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/eegdenoisenet"))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": "https://github.com/ncclabsustech/EEGdenoiseNet",
        "commit": COMMIT,
        "sampling_rate_hz": 256,
        "epoch_duration_seconds": 2,
        "files": {},
    }
    for name, expected_shape in FILES.items():
        destination = arguments.output_dir / name
        url = f"https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/{COMMIT}/data/{name}"
        if arguments.force or not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            os.replace(temporary, destination)
        values = np.load(destination, allow_pickle=False, mmap_mode="r")
        if tuple(values.shape) != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        manifest["files"][name] = {
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    with (arguments.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
