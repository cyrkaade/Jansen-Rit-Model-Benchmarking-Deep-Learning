"""Registered three-seed `b_i` robustness pilot."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
import yaml

from deepjr.baseline import (
    BaselineArrays,
    load_upstream_baseline,
    read_config,
    regression_metrics,
    split_indices,
    verify_dataset,
    write_json,
)
from deepjr.noise import ArtifactBank, apply_noise
from deepjr.published_transformer import PublishedEEGTransformer
from deepjr.reproducibility import seed_everything


def run_registered_pilot(config_path: str | Path) -> dict[str, Any]:
    config = read_config(config_path)
    _require_passed_baseline(config["baseline_gate"])
    _verify_pilot_inputs(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, original = _load_paired(config)
    split = split_indices(len(clean.target), int(config["split_seed"]))
    paired_snr = _paired_snr(clean.eeg, original.eeg)
    resolved_snr = float(np.median(paired_snr[split["train"]]))
    original_test_snr = float(np.median(paired_snr[split["test"]]))
    frozen_config = dict(config)
    frozen_config["resolved_snr_db"] = resolved_snr
    frozen_config["original_test_median_snr_db"] = original_test_snr
    with (output_dir / "frozen_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(frozen_config, stream, sort_keys=False)
    write_json(
        output_dir / "paired_data_integrity.json",
        {
            "paired_observations": len(clean.target),
            "max_abs_target_difference": float(
                np.max(np.abs(clean.target - original.target))
            ),
            "paired_original_snr_db": {
                "minimum": float(paired_snr.min()),
                "median": float(np.median(paired_snr)),
                "maximum": float(paired_snr.max()),
                "training_median": resolved_snr,
                "test_median": original_test_snr,
            },
        },
    )

    artifact_bank = ArtifactBank.from_directory(
        config["data"]["artifacts_path"],
        sampling_rate=float(config["data"]["artifact_sampling_rate"]),
        split_seed=int(config["split_seed"]),
    )
    evaluation_sets, evaluation_metadata = _evaluation_sets(
        clean,
        original,
        split["test"],
        resolved_snr,
        config,
        artifact_bank,
    )
    write_json(output_dir / "evaluation_noise_metadata.json", evaluation_metadata)

    metric_rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], tuple[Path, float, float]] = {}
    for seed in config["seeds"]:
        seed = int(seed)
        model, target_min, target_max, history, seconds = _train_condition(
            "original",
            seed,
            clean,
            original,
            split,
            resolved_snr,
            config,
            artifact_bank,
            output_dir,
        )
        model_path = output_dir / "models" / f"original_seed_{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        pd.DataFrame(history).to_csv(
            output_dir / f"training_history_original_seed_{seed}.csv", index=False
        )
        models[("original", seed)] = (model_path, target_min, target_max)
        metric_rows.extend(
            _evaluate_model(
                model,
                "original",
                seed,
                seconds,
                target_min,
                target_max,
                clean,
                split["test"],
                evaluation_sets,
                config,
                output_dir,
            )
        )
        print(f"completed original training seed {seed} in {seconds:.1f}s", flush=True)

    stronger_required, preregistered_effects = _needs_stronger_check(metric_rows)
    write_json(
        output_dir / "original_primary_effects.json",
        {
            "stronger_check_required": stronger_required,
            "effects": preregistered_effects,
        },
    )
    print(f"stronger fallback required: {stronger_required}", flush=True)
    if stronger_required:
        stronger_sets, stronger_metadata = _novel_evaluation_sets(
            clean.eeg[split["test"]],
            float(config["stronger_fallback_snr_db"]),
            config,
            artifact_bank,
            seed_offset=9000,
            severity_label="stronger_fallback",
        )
        evaluation_sets.update(stronger_sets)
        evaluation_metadata.update(stronger_metadata)
        write_json(output_dir / "evaluation_noise_metadata.json", evaluation_metadata)
        for seed in config["seeds"]:
            seed = int(seed)
            model_path, target_min, target_max = models[("original", seed)]
            model = _load_model(model_path, clean, config)
            metric_rows.extend(
                _evaluate_model(
                    model,
                    "original",
                    seed,
                    float("nan"),
                    target_min,
                    target_max,
                    clean,
                    split["test"],
                    stronger_sets,
                    config,
                    output_dir,
                )
            )

    for seed in config["seeds"]:
        seed = int(seed)
        model, target_min, target_max, history, seconds = _train_condition(
            "domain_randomized",
            seed,
            clean,
            original,
            split,
            resolved_snr,
            config,
            artifact_bank,
            output_dir,
        )
        model_path = output_dir / "models" / f"domain_randomized_seed_{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        pd.DataFrame(history).to_csv(
            output_dir / f"training_history_domain_randomized_seed_{seed}.csv",
            index=False,
        )
        metric_rows.extend(
            _evaluate_model(
                model,
                "domain_randomized",
                seed,
                seconds,
                target_min,
                target_max,
                clean,
                split["test"],
                evaluation_sets,
                config,
                output_dir,
            )
        )
        print(
            f"completed domain-randomized training seed {seed} in {seconds:.1f}s",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_tidy.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    final_effects = _decision_effects(metrics, config)
    write_json(output_dir / "decision_effects.json", final_effects)
    return {
        "resolved_snr_db": resolved_snr,
        "stronger_check_required": stronger_required,
        "metrics": metrics,
        "summary": summary,
        "decision_effects": final_effects,
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "training_condition",
        "test_noise",
        "severity_label",
        "severity_db",
        "metric",
    ]
    rows = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        values = group["value"].to_numpy(dtype=float)
        mean = float(values.mean())
        if len(values) > 1:
            std = float(values.std(ddof=1))
            margin = float(
                scipy.stats.t.ppf(0.975, len(values) - 1)
                * std
                / np.sqrt(len(values))
            )
        else:
            std = float("nan")
            margin = float("nan")
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _require_passed_baseline(path: str | Path) -> None:
    with Path(path).open(encoding="utf-8") as stream:
        status = json.load(stream)
    if status.get("status") != "passed":
        raise RuntimeError("pilot is blocked because the baseline gate did not pass")


def _verify_pilot_inputs(config: dict[str, Any]) -> None:
    data = config["data"]
    verify_dataset(data["clean_path"], data["clean_sha256"], data["clean_bytes"])
    verify_dataset(
        data["original_path"], data["original_sha256"], data["original_bytes"]
    )


def _load_paired(config: dict[str, Any]) -> tuple[BaselineArrays, BaselineArrays]:
    targets = (str(config["target"]),)
    clean = load_upstream_baseline(
        config["data"]["clean_path"], targets=targets, input_scale=1.0
    )
    original = load_upstream_baseline(
        config["data"]["original_path"], targets=targets, input_scale=1.0
    )
    ids = np.intersect1d(clean.simulation_ids, original.simulation_ids)
    clean_indices = np.searchsorted(clean.simulation_ids, ids)
    original_indices = np.searchsorted(original.simulation_ids, ids)
    return _subset(clean, clean_indices), _subset(original, original_indices)


def _subset(arrays: BaselineArrays, indices: np.ndarray) -> BaselineArrays:
    return BaselineArrays(
        eeg=arrays.eeg[indices],
        target=arrays.target[indices],
        target_names=arrays.target_names,
        simulation_ids=arrays.simulation_ids[indices],
        target_min=arrays.target[indices].min(axis=0),
        target_max=arrays.target[indices].max(axis=0),
        original_count=arrays.original_count,
    )


def _paired_snr(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    signal_power = np.mean(np.square(clean), axis=(1, 2))
    noise_power = np.mean(np.square(noisy - clean), axis=(1, 2))
    return 10.0 * np.log10(signal_power / noise_power)


def _evaluation_sets(
    clean: BaselineArrays,
    original: BaselineArrays,
    test_indices: np.ndarray,
    snr_db: float,
    config: dict[str, Any],
    artifact_bank: ArtifactBank,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paired_test_snr = float(
        np.median(_paired_snr(clean.eeg[test_indices], original.eeg[test_indices]))
    )
    sets: dict[str, dict[str, Any]] = {
        "clean": {
            "eeg": clean.eeg[test_indices],
            "severity_db": float("nan"),
            "severity_label": "clean",
        },
        "original": {
            "eeg": original.eeg[test_indices],
            "severity_db": paired_test_snr,
            "severity_label": "paired_original",
        },
    }
    metadata = {
        "clean": {"family": "clean"},
        "original": {
            "family": "original",
            "source": "historical_paired_MNE_factor_1",
            "median_actual_snr_db": paired_test_snr,
        },
    }
    novel_sets, novel_metadata = _novel_evaluation_sets(
        clean.eeg[test_indices],
        snr_db,
        config,
        artifact_bank,
        seed_offset=6800,
        severity_label="primary_snr_matched",
    )
    sets.update(novel_sets)
    metadata.update(novel_metadata)
    return sets, metadata


def _novel_evaluation_sets(
    clean_test: np.ndarray,
    snr_db: float,
    config: dict[str, Any],
    artifact_bank: ArtifactBank,
    *,
    seed_offset: int,
    severity_label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sets = {}
    metadata = {}
    noise_config = config["noise"]
    for condition_number, family in enumerate(("colored_1_f", "eog", "emg")):
        family_config = dict(noise_config)
        family_config.update({"family": family, "snr_db": snr_db})
        noisy, details = apply_noise(
            clean_test,
            family_config,
            np.random.default_rng(seed_offset + condition_number),
            artifact_bank=artifact_bank,
            artifact_split="test",
        )
        key = family if severity_label == "primary_snr_matched" else f"{family}_stronger"
        sets[key] = {
            "eeg": noisy,
            "severity_db": snr_db,
            "severity_label": severity_label,
            "test_noise": family,
        }
        metadata[key] = details
    return sets, metadata


def _train_condition(
    condition: str,
    seed: int,
    clean: BaselineArrays,
    original: BaselineArrays,
    split: dict[str, np.ndarray],
    snr_db: float,
    config: dict[str, Any],
    artifact_bank: ArtifactBank,
    output_dir: Path,
) -> tuple[PublishedEEGTransformer, float, float, list[dict[str, float]], float]:
    if condition == "original":
        train_eeg = original.eeg[split["train"]]
        validation_eeg = original.eeg[split["validation"]]
    elif condition == "domain_randomized":
        mixed_config = dict(config["noise"])
        mixed_config.update(
            {
                "family": "domain_randomized",
                "families": ["original", "colored_1_f", "eog", "emg"],
                "snr_db": snr_db,
            }
        )
        train_eeg, train_metadata = apply_noise(
            clean.eeg[split["train"]],
            mixed_config,
            np.random.default_rng(seed * 100 + 1),
            artifact_bank=artifact_bank,
            artifact_split="train",
        )
        validation_eeg, validation_metadata = apply_noise(
            clean.eeg[split["validation"]],
            mixed_config,
            np.random.default_rng(seed * 100 + 2),
            artifact_bank=artifact_bank,
            artifact_split="train",
        )
        write_json(
            output_dir / f"domain_noise_metadata_seed_{seed}.json",
            {"train": train_metadata, "validation": validation_metadata},
        )
    else:
        raise ValueError(f"unknown training condition: {condition}")

    target = clean.target[:, 0]
    target_min = float(target[split["train"]].min())
    target_max = float(target[split["train"]].max())
    return _fit_model(
        train_eeg,
        target[split["train"]],
        validation_eeg,
        target[split["validation"]],
        target_min,
        target_max,
        seed,
        config,
    )


def _fit_model(
    train_eeg: np.ndarray,
    train_target: np.ndarray,
    validation_eeg: np.ndarray,
    validation_target: np.ndarray,
    target_min: float,
    target_max: float,
    seed: int,
    config: dict[str, Any],
) -> tuple[PublishedEEGTransformer, float, float, list[dict[str, float]], float]:
    seed_everything(seed)
    training = config["training"]
    torch.set_num_threads(int(training["num_threads"]))
    device = torch.device(training["device"])
    model = PublishedEEGTransformer(
        num_channels=train_eeg.shape[1],
        num_timepoints=train_eeg.shape[2],
        output_dim=1,
        **config["model"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training["batch_size"])
    train_target_scaled = ((train_target - target_min) / (target_max - target_min)).astype(
        np.float32
    )
    validation_target_scaled = (
        (validation_target - target_min) / (target_max - target_min)
    ).astype(np.float32)
    input_scale = np.float32(config["data"]["input_scale"])
    history = []
    total_started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train_eeg), generator=generator).numpy()
        losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            eeg = torch.from_numpy(train_eeg[indices] * input_scale).to(device)
            target = torch.from_numpy(train_target_scaled[indices, None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(eeg), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = _predict_scaled(
            model, validation_eeg, input_scale, device, batch_size
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(losses)),
                "validation_mse": float(
                    np.mean(np.square(validation_prediction - validation_target_scaled))
                ),
                "seconds": time.perf_counter() - epoch_started,
            }
        )
    return model, target_min, target_max, history, time.perf_counter() - total_started


def _predict_scaled(
    model: PublishedEEGTransformer,
    eeg: np.ndarray,
    input_scale: np.float32,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            batch = torch.from_numpy(eeg[start : start + batch_size] * input_scale).to(
                device
            )
            predictions.append(model(batch).squeeze(1).cpu().numpy())
    return np.concatenate(predictions)


def _evaluate_model(
    model: PublishedEEGTransformer,
    training_condition: str,
    seed: int,
    training_seconds: float,
    target_min: float,
    target_max: float,
    clean: BaselineArrays,
    test_indices: np.ndarray,
    evaluation_sets: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    truth = clean.target[test_indices, 0]
    for key, condition in evaluation_sets.items():
        prediction_scaled = _predict_scaled(
            model,
            condition["eeg"],
            np.float32(config["data"]["input_scale"]),
            torch.device(config["training"]["device"]),
            int(config["training"]["batch_size"]),
        )
        prediction = prediction_scaled * (target_max - target_min) + target_min
        test_noise = condition.get("test_noise", key)
        raw = pd.DataFrame(
            {
                "simulation_id": clean.simulation_ids[test_indices],
                "y_true": truth,
                "y_pred": prediction,
                "seed": seed,
                "training_condition": training_condition,
                "test_noise": test_noise,
                "severity_label": condition["severity_label"],
                "severity_db": condition["severity_db"],
            }
        )
        raw_dir = output_dir / "raw_predictions"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw.to_csv(
            raw_dir / f"{training_condition}_seed_{seed}_{key}.csv", index=False
        )
        metrics = regression_metrics(truth, prediction, parameter_range=(25.0, 75.0))
        for metric, value in metrics.items():
            if metric == "n":
                continue
            rows.append(
                {
                    "seed": seed,
                    "training_condition": training_condition,
                    "test_noise": test_noise,
                    "severity_label": condition["severity_label"],
                    "severity_db": condition["severity_db"],
                    "metric": metric,
                    "value": value,
                    "training_seconds": training_seconds,
                }
            )
    return rows


def _load_model(
    path: Path, arrays: BaselineArrays, config: dict[str, Any]
) -> PublishedEEGTransformer:
    model = PublishedEEGTransformer(
        num_channels=arrays.eeg.shape[1],
        num_timepoints=arrays.eeg.shape[2],
        output_dim=1,
        **config["model"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _needs_stronger_check(
    rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, float]]:
    frame = pd.DataFrame(rows)
    pearson = frame[
        (frame.metric == "pearson")
        & (frame.training_condition == "original")
        & (frame.severity_label.isin(["paired_original", "primary_snr_matched"]))
    ]
    matched = pearson[pearson.test_noise == "original"].set_index("seed").value
    effects = {}
    meaningful = False
    for family in ("colored_1_f", "eog", "emg"):
        family_values = pearson[pearson.test_noise == family].set_index("seed").value
        drops = matched - family_values
        effects[f"{family}_mean_pearson_drop"] = float(drops.mean())
        effects[f"{family}_consistent_drop"] = bool((drops > 0).all())
        meaningful |= bool(drops.mean() >= 0.10 and (drops > 0).all())
    return not meaningful, effects


def _decision_effects(metrics: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["decision_thresholds"]
    result: dict[str, Any] = {"slices": {}, "thresholds": thresholds}
    pearson = metrics[metrics.metric == "pearson"]
    original_matched = pearson[
        (pearson.training_condition == "original")
        & (pearson.test_noise == "original")
    ].set_index("seed").value
    randomized_matched = pearson[
        (pearson.training_condition == "domain_randomized")
        & (pearson.test_noise == "original")
    ].set_index("seed").value
    result["matched_pearson_cost"] = float(
        (original_matched - randomized_matched).mean()
    )

    for severity in ("primary_snr_matched", "stronger_fallback"):
        if not (metrics.severity_label == severity).any():
            continue
        severity_result: dict[str, Any] = {}
        for family in ("colored_1_f", "eog", "emg"):
            original_pearson = pearson[
                (pearson.training_condition == "original")
                & (pearson.test_noise == family)
                & (pearson.severity_label == severity)
            ].set_index("seed").value
            randomized_pearson = pearson[
                (pearson.training_condition == "domain_randomized")
                & (pearson.test_noise == family)
                & (pearson.severity_label == severity)
            ].set_index("seed").value
            original_nrmse = metrics[
                (metrics.training_condition == "original")
                & (metrics.test_noise == family)
                & (metrics.severity_label == severity)
                & (metrics.metric == "nrmse")
            ].set_index("seed").value
            matched_nrmse = metrics[
                (metrics.training_condition == "original")
                & (metrics.test_noise == "original")
                & (metrics.metric == "nrmse")
            ].set_index("seed").value
            pearson_drop = original_matched - original_pearson
            nrmse_rise = original_nrmse / matched_nrmse - 1.0
            pearson_failure = bool(
                pearson_drop.mean() >= float(thresholds["pearson_drop"])
                and (pearson_drop > 0).all()
            )
            nrmse_failure = bool(
                nrmse_rise.mean() >= float(thresholds["nrmse_relative_rise"])
                and (nrmse_rise > 0).all()
            )
            severity_result[family] = {
                "original_mean_pearson_drop": float(pearson_drop.mean()),
                "original_pearson_drop_consistent": bool((pearson_drop > 0).all()),
                "original_mean_nrmse_relative_rise": float(nrmse_rise.mean()),
                "original_nrmse_rise_consistent": bool((nrmse_rise > 0).all()),
                "meaningful_robustness_failure": pearson_failure or nrmse_failure,
                "domain_randomization_pearson_gain": float(
                    (randomized_pearson - original_pearson).mean()
                ),
            }
        result["slices"][severity] = severity_result
    return result
