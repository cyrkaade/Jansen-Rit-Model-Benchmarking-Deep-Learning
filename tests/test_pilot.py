import numpy as np
import pandas as pd

from deepjr.pilot import _needs_stronger_check, summarize_metrics


def test_summary_reports_seed_mean_and_finite_t_interval():
    frame = pd.DataFrame(
        {
            "seed": [17, 42, 68],
            "training_condition": ["original"] * 3,
            "test_noise": ["eog"] * 3,
            "severity_label": ["primary_snr_matched"] * 3,
            "severity_db": [10.0] * 3,
            "metric": ["pearson"] * 3,
            "value": [0.7, 0.8, 0.9],
        }
    )
    summary = summarize_metrics(frame).iloc[0]
    assert summary.n_seeds == 3
    assert np.isclose(summary["mean"], 0.8)
    assert summary.ci95_low < summary["mean"] < summary.ci95_high


def test_stronger_check_follows_registered_consistent_drop_rule():
    rows = []
    for seed in (17, 42, 68):
        rows.append(_pearson_row(seed, "original", 0.9, "paired_original"))
        for family in ("colored_1_f", "eog", "emg"):
            value = 0.7 if family == "eog" else 0.88
            rows.append(_pearson_row(seed, family, value, "primary_snr_matched"))
    stronger_required, effects = _needs_stronger_check(rows)
    assert not stronger_required
    assert effects["eog_consistent_drop"]
    assert np.isclose(effects["eog_mean_pearson_drop"], 0.2)


def _pearson_row(seed, test_noise, value, severity_label):
    return {
        "seed": seed,
        "training_condition": "original",
        "test_noise": test_noise,
        "severity_label": severity_label,
        "severity_db": 10.0,
        "metric": "pearson",
        "value": value,
    }
