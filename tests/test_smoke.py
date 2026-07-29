import numpy as np

from deepjr.smoke import make_smoke_observations, run_smoke


def test_smoke_generation_is_deterministic_finite_and_shaped():
    first, first_target, first_metadata = make_smoke_observations(seed=17)
    second, second_target, second_metadata = make_smoke_observations(seed=17)
    assert first.shape == (4, 1201, 64)
    assert np.array_equal(first, second)
    assert np.array_equal(first_target, second_target)
    assert first_metadata == second_metadata
    assert np.isfinite(first).all()


def test_transformer_smoke_step_has_expected_shape_and_finite_loss():
    result = run_smoke(seed=17)
    assert result["eeg_shape"] == [4, 1201, 64]
    assert result["prediction_shape"] == [4, 1]
    assert np.isfinite(result["loss"])
