import numpy as np
import pytest

from deepjr.noise import ArtifactBank, apply_noise, resample_artifact


@pytest.fixture
def clean_eeg():
    rng = np.random.default_rng(4)
    return rng.normal(size=(3, 200, 64)).astype(np.float32)


@pytest.fixture
def artifact_bank():
    time = np.linspace(0, 2, 100, endpoint=False)
    eog = np.asarray([np.sin(2 * np.pi * (i + 1) * time / 8) for i in range(10)])
    emg = np.asarray([np.sin(2 * np.pi * (15 + i) * time) for i in range(12)])
    return ArtifactBank(eog=eog, emg=emg, sampling_rate=50.0, split_seed=68)


@pytest.mark.parametrize("family", ["original", "colored_1_f", "eog", "emg"])
def test_noise_is_deterministic_shaped_finite_and_snr_matched(
    family, clean_eeg, artifact_bank
):
    config = {"family": family, "snr_db": 3.0, "spectral_exponent": 1.0}
    first, first_metadata = apply_noise(
        clean_eeg,
        config,
        np.random.default_rng(17),
        artifact_bank=artifact_bank,
        artifact_split="train",
        sampling_rate=100.0,
    )
    second, second_metadata = apply_noise(
        clean_eeg,
        config,
        np.random.default_rng(17),
        artifact_bank=artifact_bank,
        artifact_split="train",
        sampling_rate=100.0,
    )
    assert first.shape == clean_eeg.shape
    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert np.isfinite(first).all()
    assert all(
        abs(sample["actual_snr_db"] - 3.0) < 1e-6
        for sample in first_metadata["samples"]
    )


def test_resampling_has_expected_duration():
    waveform = np.arange(512, dtype=float)
    resampled = resample_artifact(waveform, source_rate=256, target_rate=1000)
    assert len(resampled) == 2000
    assert np.isfinite(resampled).all()


def test_artifact_train_test_indices_are_disjoint_and_sampling_respects_split(
    artifact_bank,
):
    for family in ("eog", "emg"):
        train = set(artifact_bank.split_indices(family, "train"))
        test = set(artifact_bank.split_indices(family, "test"))
        assert train
        assert test
        assert train.isdisjoint(test)
        _, train_index = artifact_bank.sample(family, "train", np.random.default_rng(1))
        _, test_index = artifact_bank.sample(family, "test", np.random.default_rng(1))
        assert train_index in train
        assert test_index in test


def test_domain_randomization_is_deterministic(clean_eeg, artifact_bank):
    config = {
        "family": "domain_randomized",
        "families": ["original", "colored_1_f", "eog", "emg"],
        "snr_db": 0.0,
        "spectral_exponent": 1.0,
    }
    first, first_metadata = apply_noise(
        clean_eeg,
        config,
        np.random.default_rng(42),
        artifact_bank=artifact_bank,
        sampling_rate=100.0,
    )
    second, second_metadata = apply_noise(
        clean_eeg,
        config,
        np.random.default_rng(42),
        artifact_bank=artifact_bank,
        sampling_rate=100.0,
    )
    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
