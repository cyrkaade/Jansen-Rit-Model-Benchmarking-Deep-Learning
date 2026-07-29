# Registered pilot experiment card

## Gate and scope

The first gate is a local reproduction of strong `b_i` recovery by the
published EEG Transformer under noise factor 1.0 from MNE's ad-hoc covariance.
The historical upstream dataset and its SHA-256 are frozen in
`configs/baseline.yaml`. The novel-noise pilot in `configs/pilot.yaml` remains
blocked until the gate reaches Pearson correlation >= 0.85.

The pilot target is only `b_i`; split seed 68 is fixed and training seeds are
17, 42, and 68. Clean observations and parameter samples must be paired across
all test noise conditions. NRMSE is RMSE divided by the registered `b_i` range,
75 - 25 s^-1.

## Known deviations and ambiguities

- Python 3.12 with PyTorch 1.9, as stated in the paper, is not a supported
  combination. The compatibility environment uses current PyTorch and records
  the exact installed version.
- The upstream notebook does not seed PyTorch before its reported Transformer
  run and does not checkpoint the model or raw predictions.
- The upstream `MinMaxScaler` is fit before splitting. This target leakage is
  retained only for the exact reproduction gate and is explicitly recorded.
- Normal parameter sampling is unbounded even though the paper presents ranges.
- The notebook passes arrays as `(simulation, time, channel)` to a model whose
  variable names imply `(simulation, channel, time)`. The executed architecture
  therefore attends across 64 channel tokens while projecting 1201 time samples;
  the reproduction gate preserves that behavior despite the misleading names.
- The paper and code specify a covariance scaling factor, not an SNR-matched
  baseline severity. MNE covariance multiplication by `factor` scales noise
  variance; standard deviation changes by the square root of the factor.
- Upstream issue #2 questions the spatial validity/magnitude of the covariance
  noise. Diagnostics are required before interpreting novel noise effects.

## Decision rules

The project thresholds in `AGENTS.md` are preregistered, but all raw effects and
per-seed values must also be reported. Semi-synthetic artifacts cannot support
claims about clinical, infant, or fully empirical EEG.
