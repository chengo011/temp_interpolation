# Results: Temperature Interpolation

In the main experiment, **linear interpolation** has the lower average MAE in the direct comparison. BiLSTM: **0.3282 °C**, linear: **0.2926 °C** (equal weighting for each of the six gap lengths).

## Experimental Setup

Jena Climate, with the original ten-minute resolution. Training: 2009–2014, validation: 2015, test: 2016. There are 64 fixed validation gaps and 256 fixed test gaps per length. Each epoch draws 4096 random training windows from the entire training period. Training uses at most 50 epochs, Adam, a batch size of 64, early stopping, learning-rate reduction, and the best validation model.

Nine BiLSTM models: scenarios A and B with 3/6/12/24 hours of context on each side, plus a temperature-and-time-only model with 12 hours of context. Scenarios A and B are identical for the temperature-only model. Architecture: bidirectional LSTM (64), dropout of 0.2, bidirectional LSTM (32), dense layer (16), output (1).

All variants use the same population of complete windows and the same saved test gaps. The larger 24-hour context determines the shared eligibility of windows. Each model still receives exactly its configured context length. All models were specified before the first test evaluation and selected using validation data.

## Data Cleaning and Exclusions

420,551 original rows; 327 duplicate timestamps (first entry retained); 544 missing time steps inserted as NaN. Clearly invalid sensor values are marked as NaN. No interpolation is performed in advance. Windows with missing original values are excluded. The single observation from January 1, 2017 falls outside the configured periods.

The inputs are temperature, air pressure, relative humidity, and wind components. Potential temperature, dew point, vapor pressure variables, specific humidity, water concentration, and air density are excluded because they may mathematically encode temperature information. Daily and yearly phases are encoded cyclically, accounting for leap years. All continuous features are standardized exclusively using training statistics; masks remain binary.

## Main Experiment: MAE in °C

| gap_minutes | BiLSTM | Cubic spline | Forward fill | Linear | PCHIP |
| --- | --- | --- | --- | --- | --- |
| 10.0000 | 0.2019 | 0.0664 | 0.1534 | 0.0713 | 0.0677 |
| 30.0000 | 0.2401 | 0.1193 | 0.2734 | 0.1268 | 0.1200 |
| 60.0000 | 0.2725 | 0.2018 | 0.4530 | 0.1811 | 0.1735 |
| 120.0000 | 0.3189 | 0.3445 | 0.7884 | 0.2664 | 0.2602 |
| 240.0000 | 0.4040 | 0.6440 | 1.1845 | 0.4213 | 0.3982 |
| 360.0000 | 0.5320 | 0.9042 | 1.6825 | 0.6888 | 0.6246 |

## Relative Advantage over Linear Interpolation

Positive values indicate a lower BiLSTM MAE; negative values indicate a higher MAE.

| Gap (minutes) | BiLSTM MAE reduction (%) |
| --- | --- |
| 10.0000 | -183.3396 |
| 30.0000 | -89.3798 |
| 60.0000 | -50.4836 |
| 120.0000 | -19.7223 |
| 240.0000 | 4.1033 |
| 360.0000 | 22.7589 |

## Additional Experiments: Average BiLSTM MAE

| experiment | mae_celsius |
| --- | --- |
| A_multivariate_context_12h | 0.3282 |
| A_multivariate_context_24h | 0.3390 |
| A_multivariate_context_3h | 0.3769 |
| A_multivariate_context_6h | 0.3788 |
| A_temperature_only_context_12h | 0.4286 |
| B_multivariate_context_12h | 0.3842 |
| B_multivariate_context_24h | 0.4090 |
| B_multivariate_context_3h | 0.3821 |
| B_multivariate_context_6h | 0.4197 |

## Interpretation and Limitations

- For 10-minute gaps, the BiLSTM MAE in the main experiment is 183.3% higher than linear interpolation.
- For 30-minute gaps, the BiLSTM MAE is 89.4% higher than linear interpolation.
- For 60-minute gaps, the BiLSTM MAE is 50.5% higher than linear interpolation.
- For 120-minute gaps, the BiLSTM MAE is 19.7% higher than linear interpolation.
- For 240-minute gaps, the BiLSTM MAE is 4.1% lower than linear interpolation.
- For 360-minute gaps, the BiLSTM MAE is 22.8% lower than linear interpolation.

With 12 hours of context, the MAE is 0.3282 °C with additional sensors, 0.4286 °C with temperature and time only, and 0.3842 °C for a complete station outage. The plots also show how these differences vary by gap length.

These results apply to the fixed test gaps from 2016, the chosen architecture, and one training seed. No statistical significance tests were performed. Context windows and some test gaps may overlap, so the errors are not independent. Selecting complete windows excludes real problem intervals and limits generalizability. No comprehensive hyperparameter search was conducted. The test comparisons evaluate the predefined experimental protocol and were not used for subsequent tuning.

MAE and RMSE are calculated exclusively at masked positions in Celsius. Points receive equal weight within each gap length; the model comparison then averages the six MAE values with equal weighting. The complete CSV tables contain both metrics, gap counts, and point counts. Reconstruction examples use the first saved test gap of each length rather than a selection of particularly successful cases.

## Artifacts

- `all_test_metrics.csv`: all models, scenarios, context lengths, MAE, and RMSE.
- `sensor_availability_comparison.png/.pdf`: the effect of additional sensors and the outage scenario.
- `context_length_comparison.png/.pdf`: context comparison by scenario.
- `../../models/full_experiment/`: weights, preprocessing, configuration, training history, validation/test tables, individual predictions, and plots for each model.
- `protocol.json`: configurations and SHA-256 checksums fixed before test evaluation.

## Sources

- [TensorFlow: Jena dataset, time features, and wind preprocessing](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Original data source at the Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
- Project requirements: `docs/Anleitung.md`.
