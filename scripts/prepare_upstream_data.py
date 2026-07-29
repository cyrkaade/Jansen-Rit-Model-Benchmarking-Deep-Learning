"""Prepare hash-verified clean and historical-noise datasets from Git LFS."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


SOURCE_REF = "8f11565"
SOURCE_COMMIT = "8f11565692e229daa0b5fceb8dd4b8f0b9a246a7"
SOURCE_REPOSITORY = "lina-usc/Jansen-Rit-Model-Benchmarking-Deep-Learning"
DATASETS = {
    "clean": {
        "source": "notebooks/deepjr_training_data/xarr_all_1000.nc",
        "destination": "data/cache/upstream/xarr_all_1000.nc",
        "sha256": "8a68c3f6b57c6aac641fc3ebb0f19ba406c1426aecb55526cbc855ca12b84eee",
        "bytes": 615006452,
    },
    "original": {
        "source": "notebooks/deepjr_training_data/xarr_noise_all_1000_1.nc",
        "destination": "data/cache/upstream/xarr_noise_all_1000_1.nc",
        "sha256": "5958968acf7fd8eedec126119e27c0b396588446b5ab61fac02c610411cf4272",
        "bytes": 615006452,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    arguments = parser.parse_args()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    names = DATASETS if arguments.dataset == "all" else (arguments.dataset,)
    for name in names:
        prepare_dataset(root, name)


def prepare_dataset(root: Path, name: str) -> Path:
    specification = DATASETS[name]
    oid = str(specification["sha256"])
    destination = (root / str(specification["destination"])).resolve()
    destination.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(specification["bytes"])
    if _matches(destination, oid, expected_bytes):
        print(f"verified existing {destination} ({oid})")
        return destination

    lfs_object = _fetch_lfs_object(root, specification, oid)
    if lfs_object is not None:
        shutil.copyfile(lfs_object, destination)
    else:
        url = (
            "https://media.githubusercontent.com/media/"
            f"{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{specification['source']}"
        )
        print(f"git-lfs is unavailable; downloading {name} directly")
        _download(url, destination, expected_bytes)
    if not _matches(destination, oid, expected_bytes):
        raise ValueError(f"prepared dataset failed verification: {destination}")
    print(f"prepared {destination} ({oid})")
    return destination


def _fetch_lfs_object(
    root: Path, specification: dict[str, object], oid: str
) -> Path | None:
    if shutil.which("git-lfs") is None:
        return None
    try:
        subprocess.run(
            [
                "git",
                "lfs",
                "fetch",
                "origin",
                SOURCE_REF,
                f"--include={specification['source']}",
            ],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Git LFS fetch failed; falling back to a direct download")
        return None
    git_dir_text = subprocess.check_output(
        ["git", "rev-parse", "--git-dir"], cwd=root, text=True
    ).strip()
    git_dir = Path(git_dir_text)
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    lfs_object = git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    return lfs_object if lfs_object.is_file() else None


def _download(url: str, destination: Path, expected_bytes: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "deepjr-reproducibility-downloader"}
    )
    downloaded = 0
    with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            downloaded += len(chunk)
            print(
                f"downloaded {downloaded / (1024 ** 2):.0f} / "
                f"{expected_bytes / (1024 ** 2):.0f} MiB",
                end="\r",
                flush=True,
            )
    print()
    if downloaded != expected_bytes:
        raise ValueError(
            f"download size mismatch for {destination}: {downloaded} != {expected_bytes}"
        )
    os.replace(temporary, destination)


def _matches(path: Path, expected_hash: str, expected_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected_hash


if __name__ == "__main__":
    main()
