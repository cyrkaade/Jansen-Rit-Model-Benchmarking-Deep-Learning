import numpy as np
import pandas as pd

from deepjr.baseline import BaselineArrays
from deepjr.eog_recovery import recovery_effects, training_arrays
from deepjr.noise import ArtifactBank


def test_eog_training_arrays_are_10db_and_use_training_artifacts():
    rng = np.random.default_rng(7)
    clean = _arrays(rng.normal(size=(10, 200, 64)).astype(np.float32))
    original = _arrays(clean.eeg + rng.normal(scale=0.001, size=clean.eeg.shape))
    split = {
        "train": np.arange(0, 7),
        "validation": np.arange(7, 9),
        "test": np.arange(9, 10),
    }
    time = np.linspace(0, 2, 100, endpoint=False)
    bank = ArtifactBank(
        eog=np.asarray([np.sin(time * (index + 1)) for index in range(20)]),
        emg=np.asarray([np.cos(time * (index + 1)) for index in range(20)]),
        sampling_rate=50.0,
    )
    config = {
        "experiment_noise": {"snr_db": 10.0},
        "noise": {"eog_topography": "random_frontal", "eog_frontal_power": 3.0},
    }
    train, validation, metadata = training_arrays(
        "eog_10db", 17, clean, original, split, config, bank
    )
    assert train.shape == clean.eeg[split["train"]].shape
    assert validation.shape == clean.eeg[split["validation"]].shape
    samples = metadata["train"]["samples"] + metadata["validation"]["samples"]
    assert all(sample["family"] == "eog" for sample in samples)
    assert all(abs(sample["actual_snr_db"] - 10.0) < 1e-6 for sample in samples)
    train_artifacts = set(bank.split_indices("eog", "train"))
    assert all(sample["artifact_index"] in train_artifacts for sample in samples)


def test_recovery_effects_use_paired_seed_comparisons():
    rows = []
    for seed, control_original, control_eog, augmented_eog in (
        (17, 0.06, 0.09, 0.07),
        (42, 0.07, 0.10, 0.08),
        (68, 0.08, 0.11, 0.09),
    ):
        rows.extend(
            [
                _row(seed, "original", "original", "nrmse", control_original),
                _row(seed, "original", "eog", "nrmse", control_eog),
                _row(seed, "eog_10db", "eog", "nrmse", augmented_eog),
                _row(seed, "original", "original", "pearson", 0.96),
                _row(seed, "original", "eog", "pearson", 0.93),
                _row(seed, "eog_10db", "original", "pearson", 0.95),
                _row(seed, "eog_10db", "eog", "pearson", 0.95),
            ]
        )
    result = recovery_effects(
        pd.DataFrame(rows),
        {
            "minimum_control_nrmse_rise": 0.15,
            "recovery_fraction": 0.5,
            "max_original_pearson_cost": 0.05,
        },
    )
    assert np.isclose(result["paired_nrmse_recovery_fraction"]["mean"], 2 / 3)
    assert np.isclose(result["paired_eog_pearson_gain"]["mean"], 0.02)
    assert result["control_failure_reproduced"]
    assert result["recovery_success"]


def _arrays(eeg):
    target = np.arange(len(eeg), dtype=np.float32)[:, None]
    return BaselineArrays(
        eeg=np.asarray(eeg, dtype=np.float32),
        target=target,
        target_names=("b_i",),
        simulation_ids=np.arange(len(eeg)),
        target_min=target.min(axis=0),
        target_max=target.max(axis=0),
        original_count=len(eeg),
    )


def _row(seed, training, test, metric, value):
    return {
        "seed": seed,
        "training_condition": training,
        "test_noise": test,
        "severity_label": test,
        "severity_db": 10.0,
        "metric": metric,
        "value": value,
    }
