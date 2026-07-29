"""Deterministic, SNR-matched sensor-level noise interventions.

Canonical arrays have shape ``(observations, time, channels)`` (or ``(time,
channels)`` for one observation), matching the historical DeepJR dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class ArtifactBank:
    """EEGdenoiseNet EOG/EMG epochs with a leakage-safe deterministic split."""

    eog: np.ndarray
    emg: np.ndarray
    sampling_rate: float = 256.0
    split_seed: int = 68
    train_fraction: float = 0.8

    def __post_init__(self) -> None:
        for name, values in (("eog", self.eog), ("emg", self.emg)):
            if values.ndim != 2 or values.shape[1] < 2:
                raise ValueError(f"{name} artifacts must have shape (epochs, samples)")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} artifacts contain non-finite values")

    @classmethod
    def from_directory(cls, path: str | Path, **kwargs: Any) -> "ArtifactBank":
        root = Path(path)
        return cls(
            eog=np.load(root / "EOG_all_epochs.npy", allow_pickle=False),
            emg=np.load(root / "EMG_all_epochs.npy", allow_pickle=False),
            **kwargs,
        )

    def split_indices(self, family: str, split: str) -> np.ndarray:
        values = self._values(family)
        generator = np.random.default_rng(
            self.split_seed + (0 if family == "eog" else 1)
        )
        permutation = generator.permutation(len(values))
        boundary = int(np.floor(self.train_fraction * len(values)))
        if split == "train":
            return permutation[:boundary]
        if split == "test":
            return permutation[boundary:]
        raise ValueError("artifact split must be 'train' or 'test'")

    def sample(
        self, family: str, split: str, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        candidates = self.split_indices(family, split)
        index = int(candidates[rng.integers(0, len(candidates))])
        return np.asarray(self._values(family)[index], dtype=float), index

    def _values(self, family: str) -> np.ndarray:
        if family == "eog":
            return self.eog
        if family == "emg":
            return self.emg
        raise ValueError(f"unsupported artifact family: {family}")


def apply_noise(
    clean_eeg: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    artifact_bank: ArtifactBank | None = None,
    artifact_split: str = "train",
    sampling_rate: float = 1000.0,
    ch_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one configured noise family and return data plus audit metadata."""

    clean = np.asarray(clean_eeg, dtype=float)
    squeeze = clean.ndim == 2
    if squeeze:
        clean = clean[None, ...]
    if clean.ndim != 3:
        raise ValueError("clean_eeg must have shape (time, channels) or (sample, time, channels)")
    if not np.isfinite(clean).all():
        raise ValueError("clean_eeg contains non-finite values")

    if ch_names is None:
        ch_names = tuple(mne.channels.make_standard_montage("biosemi64").ch_names)
    if clean.shape[-1] != len(ch_names):
        raise ValueError("clean_eeg channel count does not match ch_names")

    requested_family = str(config["family"])
    supported = ("original", "colored_1_f", "eog", "emg")
    if requested_family == "clean":
        output = clean.copy()
        metadata = {
            "family": "clean",
            "samples": [{"family": "clean", "actual_snr_db": float("inf")}] * len(clean),
        }
        return (output[0] if squeeze else output), metadata

    sample_metadata = []
    noisy_samples = []
    for sample in clean:
        family = requested_family
        if family == "domain_randomized":
            choices = tuple(config.get("families", supported))
            if not choices or not set(choices).issubset(supported):
                raise ValueError("domain-randomized families are invalid")
            family = str(choices[rng.integers(0, len(choices))])

        interference, details = _make_interference(
            family,
            sample.shape,
            config,
            rng,
            artifact_bank=artifact_bank,
            artifact_split=artifact_split,
            sampling_rate=sampling_rate,
            ch_names=ch_names,
        )
        snr_db = float(config["snr_db"])
        noisy, actual_snr = _mix_at_snr(sample, interference, snr_db)
        details.update(
            {
                "family": family,
                "requested_snr_db": snr_db,
                "actual_snr_db": actual_snr,
            }
        )
        noisy_samples.append(noisy)
        sample_metadata.append(details)

    output = np.asarray(noisy_samples, dtype=np.float32)
    metadata = {"family": requested_family, "samples": sample_metadata}
    return (output[0] if squeeze else output), metadata


def resample_artifact(
    waveform: np.ndarray,
    source_rate: float,
    target_rate: float,
) -> np.ndarray:
    """Polyphase-resample one artifact waveform with rationalized rates."""

    source = int(round(source_rate))
    target = int(round(target_rate))
    divisor = int(np.gcd(source, target))
    return resample_poly(np.asarray(waveform, dtype=float), target // divisor, source // divisor)


def _make_interference(
    family: str,
    shape: tuple[int, int],
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    artifact_bank: ArtifactBank | None,
    artifact_split: str,
    sampling_rate: float,
    ch_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    n_times, n_channels = shape
    if family == "original":
        info = mne.create_info(list(ch_names), sampling_rate, ch_types="eeg")
        covariance = mne.make_ad_hoc_cov(info)
        data = np.asarray(covariance["data"], dtype=float)
        variance = data if data.ndim == 1 else np.diag(data)
        interference = rng.normal(size=shape) * np.sqrt(variance)[None, :]
        return _demean(interference), {"covariance": "mne.make_ad_hoc_cov"}

    if family == "colored_1_f":
        exponent = float(config.get("spectral_exponent", 1.0))
        white = rng.normal(size=shape)
        spectrum = np.fft.rfft(white, axis=0)
        frequencies = np.fft.rfftfreq(n_times, d=1.0 / sampling_rate)
        weights = np.zeros_like(frequencies)
        weights[1:] = frequencies[1:] ** (-exponent / 2.0)
        interference = np.fft.irfft(spectrum * weights[:, None], n=n_times, axis=0)
        return _demean(interference), {"spectral_exponent": exponent}

    if family in {"eog", "emg"}:
        if artifact_bank is None:
            raise ValueError(f"{family} noise requires an ArtifactBank")
        waveform, artifact_index = artifact_bank.sample(family, artifact_split, rng)
        waveform = resample_artifact(
            waveform, artifact_bank.sampling_rate, sampling_rate
        )
        if len(waveform) < n_times:
            repetitions = int(np.ceil(n_times / len(waveform)))
            waveform = np.tile(waveform, repetitions)
        start = int(rng.integers(0, len(waveform) - n_times + 1))
        waveform = waveform[start : start + n_times]
        waveform = waveform - waveform.mean()
        if family == "eog":
            topography, topography_name = _eog_topography(ch_names, rng, config)
        else:
            topography, topography_name = _emg_topography(ch_names, rng, config)
        interference = waveform[:, None] * topography[None, :]
        return interference, {
            "artifact_index": artifact_index,
            "artifact_split": artifact_split,
            "source_sampling_rate": artifact_bank.sampling_rate,
            "target_sampling_rate": sampling_rate,
            "crop_start": start,
            "topography": topography_name,
        }

    raise ValueError(f"unsupported noise family: {family}")


def _mix_at_snr(
    signal: np.ndarray, interference: np.ndarray, requested_snr_db: float
) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(np.square(signal)))
    interference_power = float(np.mean(np.square(interference)))
    if signal_power <= 0.0 or interference_power <= 0.0:
        raise ValueError("signal and interference must both have positive power")
    target_ratio = 10.0 ** (requested_snr_db / 10.0)
    scale = np.sqrt(signal_power / (target_ratio * interference_power))
    scaled = interference * scale
    actual_snr = 10.0 * np.log10(signal_power / np.mean(np.square(scaled)))
    return signal + scaled, float(actual_snr)


def _demean(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)


def _eog_topography(
    ch_names: tuple[str, ...],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    xyz = _positions(ch_names)
    anterior = (xyz[:, 1] - xyz[:, 1].min()) / np.ptp(xyz[:, 1])
    weights = anterior ** float(config.get("eog_frontal_power", 3.0))
    mode = str(config.get("eog_topography", "random_frontal"))
    if mode == "random_frontal":
        mode = "frontal_symmetric" if rng.random() < 0.5 else "frontal_lateralized"
    if mode == "frontal_lateralized":
        side = -1.0 if rng.random() < 0.5 else 1.0
        lateral = xyz[:, 0] / max(np.abs(xyz[:, 0]).max(), np.finfo(float).eps)
        weights *= np.clip(1.0 + side * 0.75 * lateral, 0.1, None)
    elif mode != "frontal_symmetric":
        raise ValueError(f"unsupported EOG topography: {mode}")
    return weights / np.linalg.norm(weights), mode


def _emg_topography(
    ch_names: tuple[str, ...],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    xyz = _positions(ch_names)
    candidates = tuple(config.get("emg_centers", ("T7", "T8", "P7", "P8")))
    center_name = str(candidates[rng.integers(0, len(candidates))])
    center = xyz[ch_names.index(center_name)]
    sigma = float(config.get("emg_spatial_sigma_m", 0.045))
    weights = np.exp(-np.sum(np.square(xyz - center), axis=1) / (2.0 * sigma**2))
    return weights / np.linalg.norm(weights), f"localized_{center_name}"


@lru_cache(maxsize=4)
def _positions(ch_names: tuple[str, ...]) -> np.ndarray:
    montage = mne.channels.make_standard_montage("biosemi64")
    positions = montage.get_positions()["ch_pos"]
    missing = set(ch_names) - set(positions)
    if missing:
        raise ValueError(f"channels missing from BioSemi64 montage: {sorted(missing)}")
    return np.asarray([positions[name] for name in ch_names], dtype=float)
