"""Focused severity-matched EOG augmentation experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from deepjr.baseline import (
    BaselineArrays,
    environment_record,
    read_config,
    split_indices,
    verify_dataset,
    write_json,
)
from deepjr.noise import ArtifactBank, apply_noise
from deepjr.pilot import (
    _evaluate_model,
    _fit_model,
    _load_paired,
    _paired_snr,
    summarize_metrics,
)


def run_eog_recovery(config_path: str | Path) -> dict[str, Any]:
    """Train paired control/EOG models and evaluate a fixed 10 dB EOG test set."""

    config = read_config(config_path)
    _validate_config(config)
    data = config["data"]
    verify_dataset(data["clean_path"], data["clean_sha256"], data["clean_bytes"])
    verify_dataset(
        data["original_path"], data["original_sha256"], data["original_bytes"]
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frozen_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    write_json(output_dir / "environment.json", environment_record())

    clean, original = _load_paired(config)
    split = split_indices(len(clean.target), int(config["split_seed"]))
    target_difference = float(np.max(np.abs(clean.target - original.target)))
    if target_difference != 0.0:
        raise ValueError("clean and original datasets do not have identical targets")

    artifact_bank = ArtifactBank.from_directory(
        data["artifacts_path"],
        sampling_rate=float(data["artifact_sampling_rate"]),
        split_seed=int(config["split_seed"]),
    )
    evaluation_sets, evaluation_metadata = _evaluation_sets(
        clean, original, split["test"], config, artifact_bank
    )
    write_json(output_dir / "evaluation_noise_metadata.json", evaluation_metadata)
    write_json(
        output_dir / "paired_data_integrity.json",
        {
            "paired_observations": len(clean.target),
            "max_abs_target_difference": target_difference,
            "test_observations": len(split["test"]),
            "original_test_median_snr_db": float(
                np.median(
                    _paired_snr(
                        clean.eeg[split["test"]], original.eeg[split["test"]]
                    )
                )
            ),
        },
    )

    metric_rows: list[dict[str, Any]] = []
    target = clean.target[:, 0]
    target_min = float(target[split["train"]].min())
    target_max = float(target[split["train"]].max())
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for condition in ("original", "eog_10db"):
            train_eeg, validation_eeg, metadata = training_arrays(
                condition, seed, clean, original, split, config, artifact_bank
            )
            if metadata is not None:
                write_json(
                    output_dir / f"training_noise_metadata_{condition}_seed_{seed}.json",
                    metadata,
                )
            model, _, _, history, seconds = _fit_model(
                train_eeg,
                target[split["train"]],
                validation_eeg,
                target[split["validation"]],
                target_min,
                target_max,
                seed,
                config,
            )
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"{condition}_seed_{seed}.pt")
            pd.DataFrame(history).to_csv(
                output_dir / f"training_history_{condition}_seed_{seed}.csv",
                index=False,
            )
            metric_rows.extend(
                _evaluate_model(
                    model,
                    condition,
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
            print(f"completed {condition} seed {seed} in {seconds:.1f}s", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_tidy.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    effects = recovery_effects(metrics, config["decision_thresholds"])
    write_json(output_dir / "recovery_effects.json", effects)
    return {
        "metrics": metrics,
        "summary": summary,
        "effects": effects,
        "output_dir": output_dir,
    }


def training_arrays(
    condition: str,
    seed: int,
    clean: BaselineArrays,
    original: BaselineArrays,
    split: dict[str, np.ndarray],
    config: dict[str, Any],
    artifact_bank: ArtifactBank,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Build one condition's fixed train/validation arrays without test leakage."""

    if condition == "original":
        return (
            original.eeg[split["train"]],
            original.eeg[split["validation"]],
            None,
        )
    if condition != "eog_10db":
        raise ValueError(f"unknown training condition: {condition}")

    noise_config = dict(config["noise"])
    noise_config.update(
        {
            "family": "eog",
            "snr_db": float(config["experiment_noise"]["snr_db"]),
        }
    )
    train_eeg, train_metadata = apply_noise(
        clean.eeg[split["train"]],
        noise_config,
        np.random.default_rng(seed * 100 + 1),
        artifact_bank=artifact_bank,
        artifact_split="train",
    )
    validation_eeg, validation_metadata = apply_noise(
        clean.eeg[split["validation"]],
        noise_config,
        np.random.default_rng(seed * 100 + 2),
        artifact_bank=artifact_bank,
        artifact_split="train",
    )
    return train_eeg, validation_eeg, {
        "train": train_metadata,
        "validation": validation_metadata,
    }


def recovery_effects(
    metrics: pd.DataFrame, thresholds: dict[str, float]
) -> dict[str, Any]:
    """Compute paired recovery and original-condition cost across seeds."""

    indexed = metrics.set_index(
        ["metric", "training_condition", "test_noise", "seed"]
    )["value"].sort_index()
    control_original_pearson = indexed.loc[("pearson", "original", "original")]
    control_eog_pearson = indexed.loc[("pearson", "original", "eog")]
    augmented_original_pearson = indexed.loc[("pearson", "eog_10db", "original")]
    augmented_eog_pearson = indexed.loc[("pearson", "eog_10db", "eog")]
    control_original_nrmse = indexed.loc[("nrmse", "original", "original")]
    control_eog_nrmse = indexed.loc[("nrmse", "original", "eog")]
    augmented_eog_nrmse = indexed.loc[("nrmse", "eog_10db", "eog")]

    seeds = sorted(int(seed) for seed in control_eog_nrmse.index)
    baseline_excess = control_eog_nrmse - control_original_nrmse
    baseline_relative_rise = baseline_excess / control_original_nrmse
    recovered = control_eog_nrmse - augmented_eog_nrmse
    positive_baseline_excess = baseline_excess.where(baseline_excess > 0.0)
    recovery_fraction = recovered / positive_baseline_excess
    original_pearson_cost = control_original_pearson - augmented_original_pearson
    eog_pearson_gain = augmented_eog_pearson - control_eog_pearson
    result = {
        "seeds": seeds,
        "paired_eog_pearson_gain": _series_record(eog_pearson_gain),
        "paired_eog_nrmse_reduction": _series_record(recovered),
        "control_eog_nrmse_relative_rise": _series_record(baseline_relative_rise),
        "paired_nrmse_recovery_fraction": _series_record(recovery_fraction),
        "paired_original_pearson_cost": _series_record(original_pearson_cost),
        "thresholds": thresholds,
    }
    mean_recovery = float(recovery_fraction.mean())
    mean_cost = float(original_pearson_cost.mean())
    control_failure_reproduced = bool(
        (baseline_excess > 0.0).all()
        and float(baseline_relative_rise.mean())
        >= float(thresholds["minimum_control_nrmse_rise"])
    )
    result["control_failure_reproduced"] = control_failure_reproduced
    result["recovery_success"] = bool(
        control_failure_reproduced
        and np.isfinite(mean_recovery)
        and mean_recovery >= float(thresholds["recovery_fraction"])
        and mean_cost <= float(thresholds["max_original_pearson_cost"])
    )
    return result


def _evaluation_sets(
    clean: BaselineArrays,
    original: BaselineArrays,
    test_indices: np.ndarray,
    config: dict[str, Any],
    artifact_bank: ArtifactBank,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    original_snr = float(
        np.median(_paired_snr(clean.eeg[test_indices], original.eeg[test_indices]))
    )
    noise_config = dict(config["noise"])
    snr_db = float(config["experiment_noise"]["snr_db"])
    noise_config.update({"family": "eog", "snr_db": snr_db})
    eog, metadata = apply_noise(
        clean.eeg[test_indices],
        noise_config,
        np.random.default_rng(int(config["experiment_noise"]["evaluation_seed"])),
        artifact_bank=artifact_bank,
        artifact_split="test",
    )
    return (
        {
            "original": {
                "eeg": original.eeg[test_indices],
                "severity_db": original_snr,
                "severity_label": "paired_original",
                "test_noise": "original",
            },
            "eog_10db": {
                "eeg": eog,
                "severity_db": snr_db,
                "severity_label": "eog_10db",
                "test_noise": "eog",
            },
        },
        {
            "original": {
                "family": "original",
                "source": "historical_paired_MNE_factor_1",
                "median_actual_snr_db": original_snr,
            },
            "eog_10db": metadata,
        },
    )


def _series_record(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(float)
    return {
        "per_seed": {str(int(seed)): float(value) for seed, value in clean.items()},
        "mean": float(clean.mean()) if len(clean) else float("nan"),
        "all_positive": bool((clean > 0).all()) if len(clean) else False,
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "data",
        "seeds",
        "split_seed",
        "target",
        "model",
        "training",
        "noise",
        "experiment_noise",
        "output_dir",
        "decision_thresholds",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"missing EOG recovery configuration fields: {sorted(missing)}")
    if config["target"] != "b_i":
        raise ValueError("the focused recovery experiment target must be b_i")
    if float(config["experiment_noise"]["snr_db"]) != 10.0:
        raise ValueError("the focused recovery experiment must use 10 dB EOG noise")
    if len(config["seeds"]) < 3:
        raise ValueError("the recovery experiment requires at least three seeds")
    threshold_fields = {
        "minimum_control_nrmse_rise",
        "recovery_fraction",
        "max_original_pearson_cost",
    }
    missing_thresholds = threshold_fields - set(config["decision_thresholds"])
    if missing_thresholds:
        raise KeyError(f"missing decision thresholds: {sorted(missing_thresholds)}")
