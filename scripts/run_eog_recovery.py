"""Run and plot the focused 10 dB EOG recovery experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepjr.eog_recovery import run_eog_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eog_10db.yaml")
    )
    arguments = parser.parse_args()
    result = run_eog_recovery(arguments.config)
    _plot(result["summary"], result["output_dir"] / "recovery.png")
    print(
        "10 dB EOG recovery experiment completed; "
        f"recovery success={result['effects']['recovery_success']}"
    )


def _plot(summary, path: Path) -> None:
    selected = summary[
        (summary.metric == "nrmse") & (summary.test_noise.isin(["original", "eog"]))
    ]
    conditions = ("original", "eog_10db")
    tests = ("original", "eog")
    values = np.asarray(
        [
            [
                selected[
                    (selected.training_condition == condition)
                    & (selected.test_noise == test)
                ]["mean"].iloc[0]
                for test in tests
            ]
            for condition in conditions
        ]
    )
    x = np.arange(len(tests))
    width = 0.35
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.bar(x - width / 2, values[0], width, label="Original-noise training")
    axis.bar(x + width / 2, values[1], width, label="10 dB EOG training")
    axis.set_xticks(x, ["Original test data", "10 dB EOG test"])
    axis.set_ylabel("NRMSE for b_i (lower is better)")
    axis.set_title("Severity-matched EOG augmentation")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
