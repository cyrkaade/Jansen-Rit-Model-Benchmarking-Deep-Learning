"""Run the registered baseline-gated pilot and render its primary figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepjr.pilot import run_registered_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    arguments = parser.parse_args()
    result = run_registered_pilot(arguments.config)
    output_dir = Path("results/pilot")
    _robustness_plot(result["summary"], output_dir / "robustness_primary.png")
    if result["stronger_check_required"]:
        _stronger_plot(result["summary"], output_dir / "robustness_stronger.png")
    print(
        f"pilot completed; resolved SNR={result['resolved_snr_db']:.3f} dB; "
        f"stronger check={result['stronger_check_required']}"
    )


def _robustness_plot(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[
        (summary.metric == "pearson")
        & summary.severity_label.isin(
            ["clean", "paired_original", "primary_snr_matched"]
        )
    ].copy()
    order = ["clean", "original", "colored_1_f", "eog", "emg"]
    _plot(selected, order, path, "Primary SNR-matched robustness")


def _stronger_plot(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[
        (summary.metric == "pearson")
        & summary.severity_label.isin(["paired_original", "stronger_fallback"])
    ].copy()
    order = ["original", "colored_1_f", "eog", "emg"]
    _plot(selected, order, path, "Result-triggered stronger-noise check")


def _plot(frame: pd.DataFrame, order: list[str], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(order))
    for training, marker in (("original", "o"), ("domain_randomized", "s")):
        subset = frame[frame.training_condition == training].set_index("test_noise")
        subset = subset.reindex(order)
        lower = subset["mean"] - subset["ci95_low"]
        upper = subset["ci95_high"] - subset["mean"]
        axis.errorbar(
            x,
            subset["mean"],
            yerr=np.vstack([lower, upper]),
            marker=marker,
            capsize=3,
            label=training.replace("_", " "),
        )
    axis.set_xticks(x, [name.replace("_", "\n") for name in order])
    axis.set_ylabel("Pearson correlation for b_i")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
