"""Reproduce the upstream matched-noise Transformer trend for b_i."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from deepjr.baseline import (
    environment_record,
    load_upstream_baseline,
    read_config,
    regression_metrics,
    train_baseline,
    verify_dataset,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    write_json(output_dir / "environment.json", environment_record())

    dataset = config["data"]
    try:
        verify_dataset(
            dataset["path"], dataset["expected_sha256"], dataset["expected_bytes"]
        )
        arrays = load_upstream_baseline(
            dataset["path"],
            targets=tuple(config["model_targets"]),
            input_scale=float(dataset["input_scale"]),
            clean_like_upstream=bool(dataset["clean_like_upstream"]),
        )
        started = time.perf_counter()
        model, predictions, history = train_baseline(arrays, config)
        elapsed = time.perf_counter() - started
        parameter_range = tuple(config["evaluation"]["parameter_range"])
        target = config["target"]
        metrics = regression_metrics(
            predictions[f"{target}_true"].values,
            predictions[f"{target}_pred"].values,
            parameter_range=parameter_range,
        )
        threshold = float(
            config["evaluation"]["published_qualitative_pearson_threshold"]
        )
        passed = bool(metrics["pearson"] >= threshold)

        predictions.to_csv(output_dir / "predictions.csv", index=False)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        torch.save(model.state_dict(), output_dir / "model_state.pt")
        write_json(output_dir / "metrics.json", metrics)
        write_json(
            output_dir / "status.json",
            {
                "status": "passed" if passed else "failed",
                "gate": "published_b_i_strong_recovery",
                "pearson_threshold": threshold,
                "elapsed_seconds": elapsed,
                "original_simulations": arrays.original_count,
                "cleaned_simulations": int(len(arrays.target)),
                "target_min": float(
                    arrays.target_min[arrays.target_names.index(target)]
                ),
                "target_max": float(
                    arrays.target_max[arrays.target_names.index(target)]
                ),
            },
        )
        _plot_predictions(
            predictions, metrics, target, output_dir / "predictions.png"
        )
        print(f"baseline {'PASSED' if passed else 'FAILED'}: {metrics}")
    except Exception as error:
        write_json(
            output_dir / "status.json",
            {"status": "failed", "gate": "baseline_execution", "error": repr(error)},
        )
        raise


def _plot_predictions(
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    target: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    truth = predictions[f"{target}_true"]
    predicted = predictions[f"{target}_pred"]
    axis.scatter(truth, predicted, s=18, alpha=0.75)
    low = min(truth.min(), predicted.min())
    high = max(truth.max(), predicted.max())
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="True b_i (s^-1)", ylabel="Predicted b_i (s^-1)")
    axis.set_title(f"Matched original noise, Pearson r={metrics['pearson']:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
