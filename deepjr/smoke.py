"""Fast scientific smoke path that does not require the fsaverage download."""

from __future__ import annotations

import mne
import numpy as np
import torch

from deepjr.published_transformer import PublishedEEGTransformer
from deepjr.reproducibility import seed_everything
from deepjr.simulation import (
    apply_C_factor,
    generate_stimulus,
    jr_typical_param,
    run_jr_simulation,
)


def make_smoke_observations(
    *, seed: int, sample_count: int = 4, noise_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Generate small JR observations plus MNE ad-hoc Gaussian sensor noise.

    A deterministic proxy topography replaces the expensive BEM forward model
    only in this smoke path. The neural dynamics, 64-channel montage, MNE
    covariance defaults, covariance scaling semantics, and array orientation
    are exercised; this output is not baseline evidence.
    """

    rng = np.random.default_rng(seed)
    sfreq = 1000.0
    dt = 1.0 / sfreq
    time = np.arange(0.0, 1.201, dt)
    events = np.array([[200, 0, 0]], dtype=int)
    _, pyramidal_input, inhibitory_input = generate_stimulus(
        dt, time[-1] + dt, events
    )
    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(montage.ch_names, sfreq, ch_types="eeg")
    info.set_montage(montage)
    topography = _proxy_topography(montage)
    targets = np.linspace(25.0, 75.0, sample_count, dtype=np.float32)
    clean = []

    for target in targets:
        parameters = dict(jr_typical_param)
        parameters["b_i"] = float(target)
        parameters = apply_C_factor(parameters)
        state = run_jr_simulation(
            dt,
            inhibitory_input,
            pyramidal_input,
            np.zeros_like(time),
            parameters,
        )
        source = state[1] - state[2]
        source = source - source[:200].mean()
        sensor = topography[:, None] * source[None, :]
        rms = np.sqrt(np.mean(np.square(sensor)))
        clean.append(sensor * (10e-6 / max(rms, np.finfo(float).eps)))

    clean_eeg = np.asarray(clean, dtype=np.float64)
    covariance = mne.make_ad_hoc_cov(info)
    covariance_data = np.asarray(covariance["data"], dtype=float)
    variance = (
        covariance_data
        if covariance_data.ndim == 1
        else np.diag(covariance_data)
    )
    noise = rng.normal(size=clean_eeg.shape)
    noise *= np.sqrt(variance * noise_factor)[None, :, None]
    noisy = clean_eeg + noise
    signal_power = float(np.mean(np.square(clean_eeg)))
    noise_power = float(np.mean(np.square(noise)))
    metadata: dict[str, float | str] = {
        "family": "mne_ad_hoc",
        "noise_factor": float(noise_factor),
        "snr_db": float(10.0 * np.log10(signal_power / noise_power)),
        "forward_model": "deterministic_proxy_smoke_only",
    }
    # Match the historical JRInvDataLoader/model input orientation.
    return noisy.transpose(0, 2, 1).astype(np.float32), targets, metadata


def run_smoke(seed: int = 17) -> dict[str, object]:
    seed_everything(seed)
    eeg, target, noise_metadata = make_smoke_observations(seed=seed)
    model = PublishedEEGTransformer(
        num_channels=eeg.shape[1],
        num_timepoints=eeg.shape[2],
        output_dim=1,
        embed_dim=16,
        num_heads=4,
        intermediate_dim=32,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction = model(torch.from_numpy(eeg))
    loss = torch.nn.functional.mse_loss(
        prediction.squeeze(1), torch.from_numpy(target / 75.0)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if eeg.shape != (4, 1201, 64):
        raise AssertionError(f"unexpected EEG shape: {eeg.shape}")
    if prediction.shape != (4, 1):
        raise AssertionError(f"unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(eeg).all() or not torch.isfinite(prediction).all():
        raise AssertionError("smoke path produced non-finite values")
    return {
        "seed": seed,
        "eeg_shape": list(eeg.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "noise": noise_metadata,
    }


def _proxy_topography(montage: mne.channels.DigMontage) -> np.ndarray:
    positions = montage.get_positions()["ch_pos"]
    xyz = np.asarray([positions[name] for name in montage.ch_names])
    weights = xyz @ np.array([0.35, 0.0, 1.0])
    weights -= weights.mean()
    return weights / np.linalg.norm(weights)
