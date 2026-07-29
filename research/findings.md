# Findings

## Baseline gate

Status: **passed**.

The paper's qualitative claim is strong `b_i` recovery by EEGTransformer across
noise factors, with correlations above 0.9. The only recoverable notebook output
for the matched factor-1 condition is Pearson `r=0.9272`; because training was
not seeded and predictions/checkpoints were not preserved, that number is
upstream evidence rather than a local reproduction.

An exact dependency reproduction is impossible as written: PyTorch 1.9 does not
support the paper's Python 3.12 runtime. The local gate therefore preserves the
published data, split, preprocessing, architecture, optimizer, batch size, and
50-epoch budget while using a compatible PyTorch release, recording the
deviation.

The local matched-noise gate achieved Pearson 0.88547, Spearman 0.90993, RMSE
5.04846 s^-1, NRMSE 0.10097 (RMSE divided by 50 s^-1), and MAE 2.74978 on 99
held-out observations. This exceeds the registered qualitative gate of 0.85,
although it is below the upstream notebook's unseeded 0.9272 run. The actual
sampled `b_i` values ranged 18.23-89.35 s^-1 because the upstream normal sampler
is unbounded despite the nominal 25-75 s^-1 range.

## Pilot

Status: **completed for seeds 17, 42, and 68**.

All test conditions used the same 99 clean observations and targets. The paired
original-noise training median was 50.6882 dB under the registered global-power
SNR definition; this unexpectedly weak effective noise is consistent with the
concern recorded in upstream issue #2.

Mean Pearson correlation (95% t interval across three seeds):

| Training | Test noise | Severity | Pearson mean [95% interval] |
| --- | --- | ---: | ---: |
| Original | Original paired | 51.12 dB test median | 0.96277 [0.95408, 0.97145] |
| Original | 1/f | 50.69 dB | 0.96242 [0.95363, 0.97120] |
| Original | EOG | 50.69 dB | 0.96233 [0.95353, 0.97114] |
| Original | EMG | 50.69 dB | 0.96247 [0.95369, 0.97124] |
| Domain randomized | Original paired | 51.12 dB test median | 0.96219 [0.95393, 0.97045] |
| Domain randomized | 1/f | 50.69 dB | 0.96184 [0.95356, 0.97011] |
| Domain randomized | EOG | 50.69 dB | 0.96175 [0.95342, 0.97008] |
| Domain randomized | EMG | 50.69 dB | 0.96188 [0.95359, 0.97017] |

At the primary matched severity, original-training Pearson drops were only
0.00035 (1/f), 0.00043 (EOG), and 0.00030 (EMG), and mean NRMSE changes were
below 0.5%. This is a null robustness result at the upstream effective SNR, not
evidence of robustness to typical raw artifacts.

Because no primary family crossed a threshold, the registered 10 dB check ran.
Original-training Pearson remained high: 0.95584 for 1/f, 0.94358 for EOG, and
0.95513 for EMG. However, EOG mean NRMSE increased from 0.07123 under paired
original noise to 0.08432, a mean 18.81% per-seed relative rise, with increases
for every seed. This crosses the project-specific 15% NRMSE threshold even
though the Pearson drop was only 0.01918. Neither 1/f nor EMG crossed a threshold.

Domain randomization at 50.69 dB did not recover the stronger EOG slice:
Pearson changed by only +0.00005 relative to original training, while NRMSE was
0.08317. Its matched-original Pearson cost was 0.00058, well inside the allowed
0.05. Because augmentation was trained at 50.69 dB rather than 10 dB, this does
not establish that severity-matched augmentation would fail.

## Interpretation and next decision

The primary result is null because the paper's factor-1 noise becomes roughly
51 dB after 60-trial averaging. The secondary, result-triggered slice suggests
selective sensitivity to stronger semi-synthetic EOG under the NRMSE criterion;
it does not demonstrate failure on empirical, clinical, or infant EEG.

The smallest justified next experiment is the same three seeds and frozen
`b_i` test set, training the mixed-noise model at 10 dB and evaluating only the
10 dB EOG slice plus paired original noise. This directly tests recovery at the
severity where degradation appeared, without expanding parameters, models, or
searching additional severities.
