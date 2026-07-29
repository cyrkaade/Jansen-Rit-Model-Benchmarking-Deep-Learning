"""Data, training, metrics, and provenance for the published baseline gate."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
import xarray as xr
import yaml
from sklearn.model_selection import train_test_split

from deepjr.published_transformer import PublishedEEGTransformer
from deepjr.reproducibility import seed_everything


@dataclass(frozen=True)
class BaselineArrays:
    eeg: np.ndarray
    target: np.ndarray
    target_names: tuple[str, ...]
    simulation_ids: np.ndarray
    target_min: np.ndarray
    target_max: np.ndarray
    original_count: int


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(path: str | Path, expected_sha256: str, expected_bytes: int) -> None:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"baseline dataset is missing: {dataset_path}")
    actual_bytes = dataset_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"dataset size mismatch: {actual_bytes} != {expected_bytes}")
    actual_hash = sha256_file(dataset_path)
    if actual_hash != expected_sha256:
        raise ValueError(f"dataset SHA-256 mismatch: {actual_hash}")


def load_upstream_baseline(
    path: str | Path,
    *,
    targets: tuple[str, ...] = (
        "A_e",
        "A_i",
        "b_e",
        "b_i",
        "a_1",
        "a_2",
        "a_3",
        "a_4",
        "C",
    ),
    input_scale: float = 100.0,
    clean_like_upstream: bool = True,
) -> BaselineArrays:
    """Load the historical xarray dataset and mirror upstream outlier cleaning."""

    with xr.open_dataset(path) as dataset:
        original_count = int(dataset.sizes["sim_no"])
        if clean_like_upstream:
            dataset = dataset.dropna("sim_no", subset=["evoked"])
            amplitude = np.abs(dataset["evoked"]).mean(dim=["time", "ch_names"])
            q1, q3 = amplitude.quantile([0.25, 0.75]).values
            lower = q1 - 3.0 * (q3 - q1)
            upper = q3 + 3.0 * (q3 - q1)
            keep = amplitude[(amplitude > lower) & (amplitude < upper)].sim_no
            dataset = dataset.sel(sim_no=keep)

        available = set(str(value) for value in dataset["parameters"].param.values)
        missing = set(targets) - available
        if missing:
            raise KeyError(f"targets are not in dataset parameters: {sorted(missing)}")

        # This unusual orientation exactly mirrors JRInvDataLoader in the
        # published notebook. Consequently, its attention operates across the
        # 64 channel tokens and the 1201 time samples are projected features.
        eeg = dataset["evoked"].transpose("sim_no", "time", "ch_names").values
        values = dataset["parameters"].sel(param=list(targets)).values
        simulation_ids = dataset.sim_no.values

    eeg = np.asarray(eeg, dtype=np.float32) * np.float32(input_scale)
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(eeg).all() or not np.isfinite(values).all():
        raise ValueError("cleaned baseline arrays contain non-finite values")
    return BaselineArrays(
        eeg=eeg,
        target=values,
        target_names=tuple(targets),
        simulation_ids=np.asarray(simulation_ids),
        target_min=values.min(axis=0),
        target_max=values.max(axis=0),
        original_count=original_count,
    )


def split_indices(sample_count: int, seed: int = 68) -> dict[str, np.ndarray]:
    """Mirror the paper/notebook's deterministic 80/10/10 split."""

    indices = np.arange(sample_count)
    train, temporary = train_test_split(indices, test_size=0.2, random_state=seed)
    validation, test = train_test_split(
        temporary, test_size=0.5, random_state=seed
    )
    return {"train": train, "validation": validation, "test": test}


def scale_target(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    if not np.all(maximum > minimum):
        raise ValueError("target maximum must exceed minimum")
    return (values - minimum) / (maximum - minimum)


def inverse_scale_target(
    values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    return values * (maximum - minimum) + minimum


def train_baseline(
    arrays: BaselineArrays,
    config: dict[str, Any],
) -> tuple[PublishedEEGTransformer, pd.DataFrame, list[dict[str, float]]]:
    """Train the extracted Transformer and return raw held-out predictions."""

    seed = int(config["seed"])
    split = split_indices(len(arrays.target), int(config["split_seed"]))
    seed_everything(seed)

    training = config["training"]
    torch.set_num_threads(int(training.get("num_threads", 1)))
    device = torch.device(training.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    scaled_target = scale_target(
        arrays.target, arrays.target_min, arrays.target_max
    ).astype(np.float32)
    model = PublishedEEGTransformer(
        num_channels=arrays.eeg.shape[1],
        num_timepoints=arrays.eeg.shape[2],
        output_dim=len(arrays.target_names),
        **config["model"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    loss_function = torch.nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(training["batch_size"])
    history: list[dict[str, float]] = []

    for epoch in range(int(training["epochs"])):
        started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(split["train"]), generator=generator).numpy()
        loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(permutation), batch_size):
            selection = split["train"][permutation[start : start + batch_size]]
            eeg = torch.from_numpy(arrays.eeg[selection]).to(device)
            target = torch.from_numpy(scaled_target[selection]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(eeg)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        validation_loss = _scaled_mse(
            model,
            arrays.eeg[split["validation"]],
        scaled_target[split["validation"]],
            device,
            batch_size,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_mse": loss_sum / batch_count,
                "validation_mse": validation_loss,
                "seconds": time.perf_counter() - started,
            }
        )

    predicted_scaled = _predict(
        model, arrays.eeg[split["test"]], device, batch_size
    )
    predicted = inverse_scale_target(
        predicted_scaled, arrays.target_min, arrays.target_max
    )
    predictions = pd.DataFrame(
        {"simulation_id": arrays.simulation_ids[split["test"]]}
    )
    for column, name in enumerate(arrays.target_names):
        predictions[f"{name}_true"] = arrays.target[split["test"], column]
        predictions[f"{name}_pred"] = predicted[:, column]
    return model, predictions, history


def _scaled_mse(
    model: torch.nn.Module,
    eeg: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    predicted = _predict(model, eeg, device, batch_size)
    return float(np.mean(np.square(predicted - target)))


def _predict(
    model: torch.nn.Module,
    eeg: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start : start + batch_size]).to(device)
            batches.append(model(batch).cpu().numpy())
    return np.concatenate(batches)


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    parameter_range: tuple[float, float] = (25.0, 75.0),
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - truth
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(parameter_range[1] - parameter_range[0])
    return {
        "pearson": float(scipy.stats.pearsonr(truth, prediction).statistic),
        "spearman": float(scipy.stats.spearmanr(truth, prediction).statistic),
        "rmse": rmse,
        "nrmse": rmse / denominator,
        "mae": float(np.mean(np.abs(error))),
        "n": float(len(truth)),
    }


def environment_record() -> dict[str, Any]:
    packages = {}
    for name in [
        "mne",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "xarray",
    ]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "git_revision": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
    }


def _git_output(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
