"""Render time-series, topographic, and PSD checks for every noise family."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from deepjr.baseline import load_upstream_baseline
from deepjr.noise import ArtifactBank, apply_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-db", type=float, default=50.743652)
    parser.add_argument(
        "--clean-data", type=Path, default=Path("data/cache/upstream/xarr_all_1000.nc")
    )
    parser.add_argument(
        "--artifacts", type=Path, default=Path("data/eegdenoisenet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/pilot/diagnostics")
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    clean = load_upstream_baseline(
        arguments.clean_data, targets=("b_i",), input_scale=1.0
    ).eeg[0]
    bank = ArtifactBank.from_directory(arguments.artifacts)
    montage = mne.channels.make_standard_montage("biosemi64")
    ch_names = tuple(montage.ch_names)
    info = mne.create_info(list(ch_names), 1000.0, ch_types="eeg")
    info.set_montage(montage)

    for number, family in enumerate(("original", "colored_1_f", "eog", "emg")):
        config = {
            "family": family,
            "snr_db": arguments.snr_db,
            "spectral_exponent": 1.0,
            "eog_topography": "frontal_lateralized",
            "emg_centers": ["T7", "T8", "P7", "P8"],
            "emg_spatial_sigma_m": 0.045,
        }
        noisy, _ = apply_noise(
            clean,
            config,
            np.random.default_rng(1700 + number),
            artifact_bank=bank,
            artifact_split="test",
            ch_names=ch_names,
        )
        interference = noisy - clean
        _plot_family(
            family,
            clean,
            noisy,
            interference,
            info,
            ch_names,
            arguments.snr_db,
            arguments.output_dir / f"{family}.png",
        )
    print(f"wrote diagnostics to {arguments.output_dir}")


def _plot_family(
    family: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    interference: np.ndarray,
    info: mne.Info,
    ch_names: tuple[str, ...],
    snr_db: float,
    path: Path,
) -> None:
    if family in {"eog", "emg"}:
        channel_index = int(np.argmax(np.mean(np.square(interference), axis=0)))
        channel = ch_names[channel_index]
    else:
        channel = "Cz"
        channel_index = ch_names.index(channel)
    peak = int(np.argmax(np.sqrt(np.mean(np.square(interference), axis=1))))
    frequency, power = welch(
        interference[:, channel_index], fs=1000.0, nperseg=512
    )
    time = np.arange(len(clean)) / 1000.0 - 0.2
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    window = slice(0, 600)
    axes[0].plot(time[window], clean[window, channel_index] * 1e6, label="clean")
    axes[0].plot(time[window], noisy[window, channel_index] * 1e6, label="noisy", alpha=0.8)
    axes[0].plot(
        time[window], interference[window, channel_index] * 1e6, label="noise", alpha=0.8
    )
    axes[0].set(xlabel="Time (s)", ylabel=f"{channel} (uV)")
    axes[0].legend(frameon=False, fontsize=8)
    mne.viz.plot_topomap(
        interference[peak], info, axes=axes[1], show=False, contours=0
    )
    axes[1].set_title(f"Noise topography at {time[peak]:.3f} s")
    axes[2].loglog(frequency[1:], power[1:])
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Noise PSD")
    axes[2].grid(alpha=0.2)
    figure.suptitle(f"{family}: requested SNR {snr_db:.2f} dB")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
