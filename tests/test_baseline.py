import numpy as np
import torch

from deepjr.baseline import regression_metrics, split_indices
from deepjr.published_transformer import PublishedEEGTransformer
from deepjr.transformer import EEGTransformer


def test_split_is_deterministic_disjoint_80_10_10():
    first = split_indices(1000, seed=68)
    second = split_indices(1000, seed=68)
    assert {name: len(value) for name, value in first.items()} == {
        "train": 800,
        "validation": 100,
        "test": 100,
    }
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert len(set(first["train"]) & set(first["validation"])) == 0
    assert len(set(first["train"]) & set(first["test"])) == 0
    assert len(set(first["validation"]) & set(first["test"])) == 0


def test_metrics_have_documented_range_normalized_rmse():
    truth = np.array([25.0, 50.0, 75.0])
    prediction = np.array([30.0, 55.0, 80.0])
    metrics = regression_metrics(truth, prediction, parameter_range=(25.0, 75.0))
    assert np.isclose(metrics["pearson"], 1.0)
    assert np.isclose(metrics["spearman"], 1.0)
    assert metrics["rmse"] == 5.0
    assert metrics["nrmse"] == 0.1
    assert metrics["mae"] == 5.0


def test_extracted_transformer_matches_upstream_computation():
    torch.manual_seed(11)
    upstream = EEGTransformer(
        num_channels=9,
        num_timepoints=7,
        output_dim=2,
        estim_params=("first", "second"),
        embed_dim=8,
        num_heads=2,
        intermediate_dim=16,
        ffn_output_dim=8,
        dropout=0.0,
    )
    extracted = PublishedEEGTransformer(
        num_channels=9,
        num_timepoints=7,
        output_dim=2,
        embed_dim=8,
        num_heads=2,
        intermediate_dim=16,
        dropout=0.0,
    )
    extracted_parameters = dict(extracted.named_parameters())
    with torch.no_grad():
        for name, parameter in upstream.named_parameters():
            extracted_parameters[name].copy_(parameter)
    eeg = torch.randn(3, 9, 7)
    upstream.eval()
    extracted.eval()
    assert torch.allclose(upstream(eeg), extracted(eeg), atol=1e-7, rtol=1e-6)
