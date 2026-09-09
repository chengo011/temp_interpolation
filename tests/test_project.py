import numpy as np
import pandas as pd
import pytest
import torch

from temperature_interpolation.baselines import METHODS, interpolate_temperature
from temperature_interpolation.cleaning import clean_frame, split_frame
from temperature_interpolation.config import ExperimentConfig, experiment_matrix
from temperature_interpolation.evaluation import summarize_predictions
from temperature_interpolation.features import SAFE_SENSOR_FEATURES, Standardizer, engineer_features
from temperature_interpolation.gaps import RandomGapBatches, make_manifest, mask_windows, valid_gap_starts
from temperature_interpolation.inference import interpolate_frame
from temperature_interpolation.model import build_model, masked_mse

"""Invariant-based tests for the leakage-sensitive experimental pipeline."""
@pytest.fixture
def weather_frame():
    """Create physically valid synthetic weather without consulting real test data."""
    index = pd.date_range("2014-01-01", periods=800, freq="10min")
    return pd.DataFrame({
        "T (degC)": 10 + np.sin(np.arange(800) / 20),
        "p (mbar)": 1000.0, "rh (%)": 60.0, "wv (m/s)": 3.0,
        "max. wv (m/s)": 5.0, "wd (deg)": 45.0,
        "Tpot (K)": 280.0, "rho (g/m**3)": 1200.0,
    }, index=index)


def test_cleaning_sorts_deduplicates_and_retains_missing_time():
    raw = pd.DataFrame({
        "Date Time": ["01.01.2014 00:30:00", "01.01.2014 00:00:00", "01.01.2014 00:00:00"],
        "T (degC)": [3.0, 1.0, 99.0], "wv (m/s)": [-9999.0, 2.0, 2.0],
    })
    cleaned, audit = clean_frame(raw)
    assert cleaned.index.is_monotonic_increasing and cleaned.index.is_unique
    assert audit["duplicate_timestamps"] == 1 and audit["inserted_missing_timestamps"] == 2
    assert cleaned.iloc[0]["T (degC)"] == 1.0
    assert cleaned.iloc[1:3]["T (degC)"].isna().all()
    assert np.isnan(cleaned.iloc[-1]["wv (m/s)"])


def test_chronological_split_has_no_overlap():
    index = pd.date_range("2014-12-31", "2016-01-02", freq="10min")
    frame = pd.DataFrame({"T (degC)": 10.0}, index=index)
    splits = split_frame(frame, ExperimentConfig())
    assert splits["train"].index.max() < splits["validation"].index.min()
    assert splits["validation"].index.max() < splits["test"].index.min()
    assert not set(splits["train"].index) & set(splits["test"].index)


@pytest.mark.parametrize("scenario", ["A", "B"])
@pytest.mark.parametrize("length", [1, 3, 6, 12, 24, 36])
def test_masks_truth_and_no_hidden_target(scenario, length):
    context, sensors, target = 18, 3, 0
    rng = np.random.default_rng(2)
    windows = rng.normal(size=(2, 2 * context + length, sensors + 4)).astype(np.float32)
    original = windows.copy()
    inputs, truth, mask = mask_windows(windows, context, length, sensors, target, scenario)
    np.testing.assert_array_equal(windows, original)
    np.testing.assert_array_equal(truth.numpy(), original[:, :, target])
    assert mask.sum().item() == 2 * length
    assert torch.all(inputs[:, context:context + length, target] == 0)
    assert torch.all(inputs[:, context:context + length, sensors + 4 + target] == 0)
    np.testing.assert_array_equal(inputs[:, :context, :sensors + 4].numpy(), original[:, :context])
    np.testing.assert_array_equal(inputs[:, context + length:, :sensors + 4].numpy(), original[:, context + length:])
    # Counterfactual truth must not affect any input element after masking.
    changed = original.copy()
    changed[:, context:context + length, :sensors if scenario == "B" else 1] += 1000
    changed_inputs, _, _ = mask_windows(changed, context, length, sensors, target, scenario)
    torch.testing.assert_close(inputs, changed_inputs, rtol=0, atol=0)
    if scenario == "A":
        np.testing.assert_array_equal(inputs[:, context:context + length, 1:sensors], original[:, context:context + length, 1:sensors])
    else:
        assert torch.all(inputs[:, context:context + length, :sensors] == 0)
    np.testing.assert_array_equal(inputs[:, :, sensors:sensors + 4], original[:, :, sensors:])


def test_feature_engineering_ignores_temperature_derivatives(weather_frame):
    expected = engineer_features(weather_frame, SAFE_SENSOR_FEATURES)
    altered = weather_frame.copy()
    altered["Tpot (K)"] = -100000.0
    altered["rho (g/m**3)"] = 999999.0
    pd.testing.assert_frame_equal(engineer_features(altered, SAFE_SENSOR_FEATURES), expected)
    with pytest.raises(ValueError):
        engineer_features(weather_frame, ["T (degC)", "Tpot (K)"])


def test_normalization_uses_only_training_statistics(weather_frame, tmp_path):
    training = engineer_features(weather_frame.iloc[:500], SAFE_SENSOR_FEATURES)
    validation = engineer_features(weather_frame.iloc[500:], SAFE_SENSOR_FEATURES)
    scaler = Standardizer.fit(training)
    saved_mean, saved_scale = scaler.mean.copy(), scaler.scale.copy()
    validation["T (degC)"] = 10000.0
    transformed = scaler.transform(validation)
    np.testing.assert_array_equal(scaler.mean, saved_mean)
    np.testing.assert_array_equal(scaler.scale, saved_scale)
    np.testing.assert_allclose(scaler.mean, training.mean().to_numpy())
    np.testing.assert_allclose(scaler.inverse_target(transformed[:, 0]), 10000.0, atol=0.001)
    scaler.save(tmp_path / "scaler.json")
    loaded = Standardizer.load(tmp_path / "scaler.json")
    np.testing.assert_array_equal(loaded.transform(validation), transformed)


def test_valid_windows_do_not_cross_missing_rows():
    values = np.ones((100, 3))
    values[50] = np.nan
    starts = valid_gap_starts(values, 5, 6)
    assert starts.min() >= 5 and starts.max() + 6 + 5 <= len(values)
    assert all(not (start - 5 <= 50 < start + 6 + 5) for start in starts)


def test_fixed_gap_manifest_is_reproducible(weather_frame):
    features = engineer_features(weather_frame, SAFE_SENSOR_FEATURES)
    first = make_manifest(features, [1, 6, 36], 144, 12, 7)
    second = make_manifest(features, [1, 6, 36], 144, 12, 7)
    pd.testing.assert_frame_equal(first, second)
    assert not first.gap_id.duplicated().any()
    for row in first.itertuples():
        assert row.gap_start in valid_gap_starts(features.to_numpy(), 144, row.gap_length)


def test_random_training_batches_are_reproducible_and_lengths_vary():
    config = ExperimentConfig(context_steps=18, samples_per_epoch=1024)
    values = np.ones((1000, 11), dtype=np.float32)
    sampler = RandomGapBatches(values, config)
    first, second = list(sampler.epoch(1)), list(sampler.epoch(1))
    lengths = set()
    for batch_one, batch_two in zip(first, second, strict=True):
        for one, two in zip(batch_one, batch_two, strict=True):
            torch.testing.assert_close(one, two)
        lengths.add(int(batch_one[2][0].sum()))
    assert len(lengths) > 1


@pytest.mark.parametrize("context,length", [(18, 1), (36, 3), (72, 36), (144, 24)])
def test_model_outputs_one_value_per_timestamp(context, length):
    torch.set_num_threads(2)
    config = ExperimentConfig(context_steps=context)
    model = build_model(config, 18)
    inputs = torch.randn(2, context * 2 + length, 18)
    assert model(inputs).shape == inputs.shape[:2]


def test_loss_ignores_unmasked_values_and_gradients():
    predictions = torch.tensor([[999.0, 2.0, 4.0, -999.0]], requires_grad=True)
    truth = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    loss = masked_mse(predictions, truth, mask)
    assert loss.item() == 5.0
    loss.backward()
    assert predictions.grad[0, 0].item() == predictions.grad[0, -1].item() == 0.0
    with pytest.raises(ValueError):
        masked_mse(predictions, truth, torch.zeros_like(mask))


@pytest.mark.parametrize("method", ["Linear", "Cubic spline", "PCHIP"])
def test_interpolation_is_exact_on_a_straight_line(method):
    truth = np.arange(10, dtype=float) * 2 + 3
    damaged = truth.copy()
    mask = np.zeros(10, dtype=bool)
    mask[3:7] = True
    damaged[mask] = np.nan
    np.testing.assert_allclose(interpolate_temperature(damaged, mask, method), truth[mask], atol=1e-12)
    damaged[mask] = 100000.0
    np.testing.assert_allclose(interpolate_temperature(damaged, mask, method), truth[mask], atol=1e-12)


def test_forward_fill_and_no_extrapolation():
    np.testing.assert_array_equal(interpolate_temperature([1, np.nan, np.nan, 4], [0, 1, 1, 0], "Forward fill"), [1, 1])
    with pytest.raises(ValueError):
        interpolate_temperature([np.nan, 2, 3], [1, 0, 0], "Linear")


def test_metrics_are_computed_in_supplied_celsius_units():
    points = pd.DataFrame({"method": ["Linear"] * 2, "gap_length": [1, 1], "gap_id": ["a", "b"],
                           "true_temperature": [10.0, 20.0], "predicted_temperature": [11.0, 23.0]})
    metrics = summarize_predictions(points).iloc[0]
    assert metrics.mae_celsius == 2.0
    assert metrics.rmse_celsius == pytest.approx(np.sqrt(5))


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_real_gap_inference_preserves_observed_values(weather_frame, scenario):
    config = ExperimentConfig(context_steps=18, scenario=scenario)
    scaler = Standardizer.fit(engineer_features(weather_frame, config.sensor_features))
    model = build_model(config, 18)
    damaged = weather_frame.copy()
    damaged.iloc[100:106, damaged.columns.get_loc(config.target)] = np.nan
    if scenario == "B":
        damaged.iloc[100:106, :] = np.nan
    original = damaged.copy(deep=True)
    result = interpolate_frame(damaged, model, scaler, config)
    pd.testing.assert_frame_equal(damaged, original)
    assert result.is_reconstructed.sum() == 6
    assert result.reconstructed_temperature.notna().all()
    np.testing.assert_array_equal(result.loc[~result.is_reconstructed, "reconstructed_temperature"],
                                  weather_frame.loc[~result.is_reconstructed, config.target])
    assert result[config.target].iloc[100:106].isna().all()


def test_inference_rejects_missing_context(weather_frame):
    config = ExperimentConfig(context_steps=18)
    scaler = Standardizer.fit(engineer_features(weather_frame, config.sensor_features))
    damaged = weather_frame.copy()
    damaged.iloc[0, damaged.columns.get_loc(config.target)] = np.nan
    with pytest.raises(ValueError, match="context"):
        interpolate_frame(damaged, build_model(config, 18), scaler, config)


def test_experiment_matrix_matches_requested_comparisons():
    configs = experiment_matrix()
    assert len(configs) == 9
    for scenario in ["A", "B"]:
        assert {config.context_steps for config in configs if config.scenario == scenario} == {18, 36, 72, 144}
    for config in configs:
        config.validate()


def test_small_model_can_learn_a_masked_reconstruction():
    """Exercise actual backpropagation without touching validation or test data."""
    torch.manual_seed(1)
    torch.set_num_threads(2)
    config = ExperimentConfig(hidden_sizes=[4, 3], dense_size=4, dropout=0.0)
    model = build_model(config, 3)
    inputs = torch.zeros(8, 7, 3)
    truth = torch.ones(8, 7) * 0.4
    mask = torch.zeros_like(truth)
    mask[:, 2:5] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = masked_mse(model(inputs), truth, mask).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = masked_mse(model(inputs), truth, mask)
        loss.backward()
        optimizer.step()
    assert masked_mse(model(inputs), truth, mask).item() < initial * 0.1
