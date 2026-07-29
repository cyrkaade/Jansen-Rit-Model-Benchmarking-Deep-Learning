"""Prepare hash-verified clean and historical-noise datasets from Git LFS."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


SOURCE_REF = "8f11565"
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
    subprocess.run(["git", "lfs", "version"], cwd=root, check=True)
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
    git_dir_text = subprocess.check_output(
        ["git", "rev-parse", "--git-dir"], cwd=root, text=True
    ).strip()
    git_dir = Path(git_dir_text)
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    oid = str(specification["sha256"])
    lfs_object = git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not lfs_object.is_file():
        raise FileNotFoundError(f"Git LFS object was not downloaded: {lfs_object}")

    destination = (root / str(specification["destination"])).resolve()
    destination.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _matches(destination, oid, int(specification["bytes"])):
        shutil.copyfile(lfs_object, destination)
    if not _matches(destination, oid, int(specification["bytes"])):
        raise ValueError(f"prepared dataset failed verification: {destination}")
    print(f"prepared {destination} ({oid})")
    return destination


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
