from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .cleaning import clean_frame
from .config import ExperimentConfig
from .features import Standardizer, engineer_features
from .training import load_trained_model, set_reproducibility

"""Apply a saved model to bounded real gaps while preserving observed values."""

def interpolate_frame(frame, model, standardizer, config):
    """Fill existing target gaps and explicitly label every reconstructed value.

    This API expects a cleaned, sorted ten-minute grid. It refuses extrapolation,
    unsupported gap lengths, incomplete context, and extra missing sensors in
    scenario A. These conditions differ from the model's training distribution.
    Original observations are never replaced by model outputs.
    """
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Input timestamps must be sorted and unique.")
    if not (frame.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all():
        raise ValueError("Input must have a complete ten-minute timestamp grid.")
    features = engineer_features(frame, config.sensor_features)
    normalized = standardizer.transform(features)
    target_index = config.sensor_features.index(config.target)
    missing = frame[config.target].isna().to_numpy()
    edges = np.diff(np.concatenate(([False], missing, [False])).astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = frame.copy(deep=True)
    reconstructed = np.zeros(len(frame), dtype=bool)
    estimates = result[config.target].to_numpy(copy=True)
    model.eval()
    for start, end in zip(starts, ends, strict=True):
        length = end - start
        if int(length) not in config.gap_lengths:
            raise ValueError(f"Gap length {length} was not included in training: {config.gap_lengths}.")
        left, right = start - config.context_steps, end + config.context_steps
        if left < 0 or right > len(frame):
            raise ValueError("Each gap needs the configured amount of context on both sides.")
        window = normalized[left:right].copy()
        sensor_count = len(config.sensor_features)
        mask = np.ones((len(window), sensor_count), dtype=np.float32)
        gap = slice(config.context_steps, config.context_steps + length)
        if config.scenario == "A":
            window[gap, target_index] = 0.0
            mask[gap, target_index] = 0.0
        else:
            window[gap, :sensor_count] = 0.0
            mask[gap, :] = 0.0
        if not np.isfinite(window).all():
            raise ValueError("An observed input or required context is missing or invalid.")
        inputs = torch.from_numpy(np.concatenate((window, mask), axis=-1)[None].astype(np.float32))
        with torch.inference_mode():
            predicted = model(inputs).numpy()[0, gap]
        estimates[start:end] = standardizer.inverse_target(predicted, config.target)
        reconstructed[start:end] = True
    result["reconstructed_temperature"] = estimates
    result["is_reconstructed"] = reconstructed
    return result


def interpolate_csv(input_path: Path, output_path: Path, model_dir: Path):
    """Load Jena-style CSV data and write truth-preserving reconstruction columns."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be a separate file to preserve the original observations.")
    config = ExperimentConfig.load(model_dir / "config.json")
    set_reproducibility(config.seed, config.threads)
    frame, audit = clean_frame(pd.read_csv(input_path))
    standardizer = Standardizer.load(model_dir / "preprocessing.json")
    model = load_trained_model(model_dir, config)
    result = interpolate_frame(frame, model, standardizer, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index_label="timestamp")
    from .pipeline import write_json
    write_json(output_path.with_suffix(".audit.json"), audit)
    return result
