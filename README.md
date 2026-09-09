# Temperature Interpolation with Deep Learning

This Python project reconstructs **internal data gaps** in the Jena Climate Dataset
2009–2016. It explicitly uses observations **before and after** a gap.
The target variable is `T (degC)` and the temporal resolution is ten minutes.

## Results and Files

The full experiment suite is complete: nine trained models,
both outage scenarios and all agreed context comparisons. All 34 automated
tests and the independent final verification of the saved results have passed.

In the main experiment, classical methods are more accurate for short gaps. For
six-hour gaps the BiLSTM MAE is 0.5320 °C compared with 0.6888 °C for linear
interpolation, a reduction of 22.8%. With equal weighting across all six gap lengths,
PCHIP achieves approximately 0.274 °C and is more accurate overall than the BiLSTM at 0.328 °C.
These findings apply to the fixed 2016 test cases and the training seed used in this run.

The results are available here:

- [Results report](results/full_experiment/Results_Documentation.md)
- [All test metrics: MAE and RMSE](results/full_experiment/all_test_metrics.csv)
- [Sensor availability comparison](results/full_experiment/sensor_availability_comparison.png)
- [Context length comparison](results/full_experiment/context_length_comparison.png)
- [Main model and associated artifacts](models/full_experiment/A_multivariate_context_12h/)
- [Automated tests](results/tests.xml)
- [Independent final verification](results/full_experiment/artifact_verification.json)

A fully completed run creates `results/full_experiment/complete.json`.
If this file is missing, the full experiment suite has not yet finished.
The project does not assume that deep learning will outperform classical methods.
The results report is generated from the actual test metrics.

## Installation

A fresh installation requires Python 3.12. Run all commands from the
`temp_interpolation` project folder. This project copy already has a
local Python environment under `.venv`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r requirements.txt
```

To repeat the experiment with the same library versions use the dependency
file saved during the run instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache\pip -r results\full_experiment\requirements.lock.txt
```

PyTorch uses an available CUDA GPU or the CPU otherwise. Model selection,
training settings and the number of experiments do not depend on the device.
Bit-for-bit identical results across different hardware and library versions
are not guaranteed; random seeds are set and deterministic PyTorch operations are enabled.

## Data Preparation

```powershell
.\.venv\Scripts\python.exe run_project.py prepare
```

The first invocation downloads the original ZIP archive from the TensorFlow data source.
The download requires internet access. Subsequent runs reuse the local file.
SHA-256 checksums identify the exact files used.

Preparation includes:

1. Parse timestamps, sort them stably and handle duplicates by keeping the first record.
2. Check ten-minute intervals and insert NaN rows for missing timestamps.
3. Mark non-finite and physically implausible values as NaN including `-9999` wind values.
4. Split chronologically: training 2009–2014, validation 2015 and test 2016.
5. Identify a shared set of complete windows and save fixed comparison gaps.

The source file contains a final observation on January 1, 2017. This falls
outside the configured periods. Originally missing values are **not** interpolated
before the experiment. Window eligibility is checked using all
allowed sensors and the largest context including for shorter contexts
and the temperature-only model. This keeps the comparison cases identical.

The original timestamps have no explicit time zone. Their time labels are
preserved; no speculative time-zone or daylight-saving correction is applied.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/tests.xml
```

The tests check chronological ordering and separation, masking in both scenarios,
preservation of ground truth, absence of hidden target information,
output dimensions, masked loss and its gradients, training-only normalization,
reproducible gaps, mathematically correct interpolation, actual
backpropagation and application to new gaps.

After a complete run, the saved model checkpoints, test positions,
Celsius reference values and metrics can also be verified independently:

```powershell
.\.venv\Scripts\python.exe scripts\verify_artifacts.py
```

## Training, Evaluation, and Plots

```powershell
.\.venv\Scripts\python.exe -u run_project.py run --run-name full_experiment
```

This command executes the entire experimental protocol:

1. Prepare the data and evaluate classical validation baselines.
2. Train and validate all nine BiLSTM models.
3. Load the best model for each run, selected exclusively by validation loss.
4. Only then evaluate all models on the previously unused test year.
5. Generate result tables, individual predictions, plots and the results report.

The project uses linear interpolation, cubic splines (`not-a-knot`), PCHIP and
forward fill as baselines. They receive only observed temperature values;
the hidden original values are physically removed before calling each method.

The BiLSTM consists of a bidirectional LSTM with 64 units, dropout of 0.2, a second
bidirectional LSTM with 32 units and dense layers with 16 and one unit, respectively.
It outputs a temperature at every time step. MSE is computed exclusively at
artificially hidden positions. The input also contains
a binary observation mask for each sensor used.

Default settings: Adam, a learning rate of 0.001, a batch size of 64, at most 50 epochs,
early stopping after seven epochs without improvement and halving the learning rate
on a plateau. Each epoch draws 4096 windows with random gap lengths from the entire
training period. An epoch therefore does not cover every possible,
heavily overlapping window. A gap length is randomly selected per
batch so that sequences of different lengths do not require padding.

Each gap length has 64 fixed validation gaps and 256 fixed test gaps.
The six lengths are 1/3/6/12/24/36 time steps = 10/30/60/120/240/360 minutes.
All methods and models use exactly the same gap positions.

### Full Experiment Matrix

| Variant | Scenario | Context on each side |
| --- | --- | --- |
| Multiple sensors, main experiment | A: only temperature is missing | 12 hours |
| Temperature + time only | A, identical to B with this feature selection | 12 hours |
| Multiple sensors | B: all sensors are missing | 12 hours |
| Multiple sensors | A and B, trained separately | 3 hours |
| Multiple sensors | A and B, trained separately | 6 hours |
| Multiple sensors | A and B, trained separately | 24 hours |

In scenario B, only time features remain available during the gap.
The optional Transformer and optional gaps longer than six hours are not part
of the agreed experimental protocol.

### Configuration and Repeated Runs

The `configs/` directory contains one JSON file per experiment. These files allow
changes to features, time periods, gap lengths, context, batch size, epochs,
learning rate, LSTM sizes, dropout, seed, and scenario. Shared settings for the
time split, gaps and seed must agree across all nine files. When changing the
experimental protocol, first back up the old files from `data/processed/` in another
project subfolder, because conflicting saved comparison gaps are deliberately
not overwritten silently.

Use a new name for a new run:

```powershell
.\.venv\Scripts\python.exe -u run_project.py run --run-name repeat_01
```

The saved `protocol.json` contains code, data, and manifest checksums, as well as
all settings. An existing run is not mixed with changed code or settings.
Completed training runs are reused when the protocol matches. A run interrupted
**during training** is not resumed from an incomplete state, a new run name
is required. Re-evaluating a completed run loads the same
best model weights.

## Preventing Data Leakage

Allowed sensor inputs are temperature, air pressure, relative humidity and the
components of wind velocity and maximum wind velocity. Wind direction is converted
into components using sine and cosine rather than being supplied as a raw
angle. Daily and yearly phases are encoded cyclically. The yearly phase
accounts for leap years.

Potential temperature, dew point, vapor pressure variables, specific humidity,
water concentration and air density are excluded as a precaution: combinations
of these quantities can mathematically encode temperature information. The reasons
are also saved in each `preprocessing.json` file.

Means and standard deviations are calculated from training data only.
Missing model inputs are set to zero after normalization and explicitly marked
as missing by the binary masks. The complete target values are copied before
masking and are not passed along as an additional feature.
No windows overlap across the chronological data splits.

## Interpolating New Data

A new CSV file uses the original dataset format:

- A `Date Time` column formatted as `dd.mm.yyyy HH:MM:SS`.
- A target column named `T (degC)`; missing values are left blank.
- For the multivariate model, also include `p (mbar)`, `rh (%)`, `wv (m/s)`,
  `max. wv (m/s)`, and `wd (deg)`.
- A ten-minute grid and sufficient complete context on both sides.

```powershell
.\.venv\Scripts\python.exe run_project.py interpolate --input examples\input_scenario_A.csv --output examples\reconstructed_scenario_A.csv --model-dir models\full_experiment\A_multivariate_context_12h
```

For a complete station outage, select `B_multivariate_context_12h` instead.
A temperature-only model requires only the date and temperature. In scenario A,
the additional sensors must remain available inside the gap. In scenario B,
all sensor columns are masked within the temperature gap, even if some
of them happen to contain values.

The output CSV contains the cleaned target column, the additional
`reconstructed_temperature` column, and the `is_reconstructed` flag. Valid observed
values are preserved; the original file is not overwritten.
A companion `*.audit.json` file documents the cleaning process.

Only gap lengths included in training are accepted. Missing context,
gaps at the boundaries, other gaps within the required context and incomplete
required sensor inputs are rejected with an explanatory error message. The
system does not extrapolate beyond the available time interval.

## Metrics and Interpretation

MAE and RMSE are calculated in °C after inverse transformation, exclusively for
artificially hidden values. Tables contain one row per method and gap length,
along with the number of gaps and points. The average model MAE in the
bar chart weights each gap length equally so that long gaps do not dominate
simply because they contain more points.

The saved gaps may overlap, their errors are not independent.
The results apply to the selected complete windows from 2016 and
one training seed. They do not establish statistical significance or
guarantee performance for other stations or outage conditions. No
hyperparameters are optimized using the test dataset.

Reconstruction plots always use the first saved gap of each length.
Examples are not selected based on model performance. All plots
are saved as PNG and PDF files.

## Project Structure

```text
temp_interpolation/
  configs/                    central JSON configurations
  data/raw/                   original download, CSV, provenance, and checksums
  data/processed/             cleaned data, audit, splits, and fixed gaps
  examples/                   input and reconstruction examples
  models/full_experiment/     nine models with configurations and results
  results/full_experiment/    overall comparison, report, and reproducibility protocol
  tests/                      automated invariant and functional tests
  temperature_interpolation/
    config.py                 configuration and experiment matrix
    download.py               original data download
    cleaning.py               cleaning and chronological splitting
    features.py               feature engineering and normalization
    gaps.py                   gaps, masks, and data batches
    baselines.py              classical interpolation methods
    model.py                  BiLSTM and masked loss
    training.py               training and best checkpoints
    evaluation.py             MAE/RMSE and individual predictions
    visualization.py          scientific results plots
    inference.py              reconstruction of new data gaps
    pipeline.py               ordered end-to-end workflow
  run_project.py              entry point
  requirements.txt            library requirements
  .venv/                     local Python environment
  .cache/                    local download, test, and plot caches
```

## Sources

- [Official TensorFlow tutorial with the Jena data source and preprocessing](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Jena weather data from the Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
