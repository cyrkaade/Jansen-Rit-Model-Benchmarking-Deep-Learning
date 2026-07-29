# Run log

## 2026-07-29 - baseline audit and implementation

- Read `AGENTS.md`, repository history/status, README, requirements, simulation
  and Transformer code, relevant notebook cells/outputs, and open upstream
  issues #1 and #2.
- Verified arXiv v2 (18 March 2025) is the latest authoritative paper version.
- Visually inspected paper pages 3-7 and reviewed simulation, BEM forward model,
  MNE ad-hoc covariance scaling, 80/10/10 split (random state 68), Transformer,
  results, limitations, and future work.
- Found the notebook's matched-noise result: one unseeded run reports Pearson
  `r=0.9272` for `b_i` on its 10% test split. The notebook also preserves a
  prior Transformer backward-pass failure.
- Located historical Git-LFS baseline dataset pointer: 615,006,452 bytes,
  SHA-256 `5958968acf7fd8eedec126119e27c0b396588446b5ab61fac02c610411cf4272`.
- Runtime audit: Windows, Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA GPU exposed,
  AMD Ryzen 7 8845HS (8 cores/16 threads). MNE was not initially installed.
- Added deterministic baseline modules, configs, CLI scripts, smoke/unit tests,
  and a compatibility environment specification. No novel-noise experiment has
  run; `configs/pilot.yaml` remains baseline-gated.

Commands issued for the reproducible path:

```powershell
uv venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements-repro.txt
uv pip install --python .venv/Scripts/python.exe -e . --no-deps
powershell -ExecutionPolicy Bypass -File scripts/prepare_upstream_baseline.ps1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/smoke_test.py --seed 17
.venv/Scripts/python.exe scripts/reproduce_baseline.py --config configs/baseline.yaml
```

Execution outcomes:

- Isolated environment installed successfully with Python 3.13.5, MNE 1.9.0,
  and PyTorch 2.10.0+cpu. The first test collection exposed missing explicit
  `seaborn`/`frozendict` dependencies; the environment file was corrected.
- Final tests: `15 passed` (only upstream Matplotlib/pyparsing deprecation
  warnings). `compileall` and `git diff --check` also passed.
- Smoke test passed for `(4, 1201, 64)` deterministic observations, finite
  predictions, one backward/optimizer step, and MNE ad-hoc noise.
- Baseline dataset was recovered from historical Git LFS and verified at
  615,006,452 bytes with the registered SHA-256. Cleaning retained 984/1000
  simulations.
- Baseline gate passed in 85.57 s: Pearson 0.88547, Spearman 0.90993, RMSE
  5.04846 s^-1, NRMSE 0.10097, MAE 2.74978, n=99.
- Recovered and verified the paired clean dataset. All 984 retained simulation
  IDs had identical parameters across clean/original data. Original paired
  sensor-noise SNR ranged 26.40-58.58 dB; the training median 50.6882 dB was
  frozen before pilot fitting.
- Downloaded EEGdenoiseNet EOG/EMG arrays from pinned commit
  `8d290661146c7189c98cc04812d37371d4b9426c`. Verified EOG shape `(3400,512)`,
  SHA-256 `fe31f2c22efce5da7488a1047b6f05d678e80f48d22e20a70972aaf13784fa0a`;
  EMG shape `(5598,512)`, SHA-256
  `8bb16dff87582823488dd2189dba9d8fa019a8cbc9af9a8a19acb421b3fcfb16`.
  Train/test artifact counts were 2720/680 EOG and 4478/1120 EMG.
- Rendered and visually inspected time-series, topography, and PSD diagnostics
  for original, 1/f, EOG, and EMG at 50.6882 dB and the registered 10 dB
  fallback.
- Pilot completed without stderr. The six 50-epoch fits took 533.46 s total
  training time (86.04-90.58 s each) on CPU. Peak native memory was not sampled;
  observed process working set during monitoring was approximately 1.4-1.6 GB.
- The primary 50.6882 dB slices did not cross a robustness-failure threshold.
  This triggered the preregistered 10 dB check. At 10 dB, only EOG crossed a
  project threshold: original-training mean NRMSE rose 18.81% relative to paired
  original noise, consistently across all seeds. Pearson fell only 0.01918.
- Domain randomization trained at the primary 50.6882 dB severity did not recover
  the 10 dB EOG effect (Pearson gain +0.00005); it cost 0.00058 Pearson on matched
  original noise. No five-seed expansion was run because recovery was absent.

Additional completed commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_upstream_clean.ps1
.venv/Scripts/python.exe scripts/download_artifacts.py --output-dir data/eegdenoisenet
.venv/Scripts/python.exe scripts/noise_diagnostics.py --snr-db 50.688213 --output-dir results/pilot/diagnostics_primary
.venv/Scripts/python.exe scripts/noise_diagnostics.py --snr-db 10 --output-dir results/pilot/diagnostics_stronger
.venv/Scripts/python.exe scripts/run_pilot.py --config configs/pilot.yaml
```
