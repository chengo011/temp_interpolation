import numpy as np
import pandas as pd
import torch

"""Fixed evaluation gaps and random training batches with explicit missing masks."""

def valid_gap_starts(values: np.ndarray, context: int, gap_length: int):
    """Find gaps whose complete window has finite original observations.

    A prefix sum counts bad rows in constant time per candidate. This rejects
    windows containing inserted timestamps, invalid sensors, or unknown truth.
    It must be called independently on each chronological split.
    """
    window_length = 2 * context + gap_length
    if len(values) < window_length:
        return np.empty(0, dtype=np.int64)
    bad_rows = (~np.isfinite(values).all(axis=1)).astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(bad_rows)))
    bad_counts = cumulative[window_length:] - cumulative[:-window_length]
    return np.flatnonzero(bad_counts == 0) + context


def make_manifest(features, gap_lengths, max_context, count_per_length, seed):
    """Save random evaluation gaps from the common largest-context population.

    Every model receives these same positions. Context windows may overlap;
    examples are therefore not statistically independent. Positions are sampled
    without replacement within each length, with no selection by model error.
    """
    rng = np.random.default_rng(seed)
    records = []
    for length in gap_lengths:
        candidates = valid_gap_starts(features.to_numpy(), max_context, length)
        if len(candidates) < count_per_length:
            raise ValueError(f"Only {len(candidates)} eligible gaps for length {length}.")
        positions = np.sort(rng.choice(candidates, count_per_length, replace=False))
        for position in positions:
            records.append({
                "gap_id": f"g{length:02d}_{position:06d}",
                "gap_start": int(position), "gap_length": int(length),
                "timestamp": str(features.index[position]),
            })
    return pd.DataFrame(records)


def mask_windows(windows, context, gap_length, sensor_count, target_index, scenario):
    """Separate immutable truth from damaged inputs and per-sensor masks.

    Numerical input zero means the TRAINING mean. A separate mask distinguishes
    it from an actual observation. Time features remain visible in both
    scenarios because station downtime does not hide the time of day.
    """
    if scenario not in {"A", "B"}:
        raise ValueError("Scenario must be A or B.")
    if windows.ndim != 3 or not np.isfinite(windows).all():
        raise ValueError("Training/evaluation windows must be finite three-dimensional arrays.")
    if windows.shape[1] != 2 * context + gap_length:
        raise ValueError("Window does not have exactly the requested context on each side.")
    truth = windows[:, :, target_index].copy()
    damaged = windows.copy()
    observed = np.ones((*windows.shape[:2], sensor_count), dtype=np.float32)
    loss_mask = np.zeros(windows.shape[:2], dtype=np.float32)
    gap = slice(context, context + gap_length)
    if scenario == "A":
        damaged[:, gap, target_index] = 0.0
        observed[:, gap, target_index] = 0.0
    else:
        damaged[:, gap, :sensor_count] = 0.0
        observed[:, gap, :] = 0.0
    loss_mask[:, gap] = 1.0
    inputs = np.concatenate((damaged, observed), axis=-1).astype(np.float32)
    return tuple(torch.from_numpy(array) for array in (inputs, truth, loss_mask))


def gather_batch(values, starts, gap_length, config):
    """Extract equally sized windows and mask their center without modifying data."""
    offsets = np.arange(-config.context_steps, gap_length + config.context_steps)
    windows = values[np.asarray(starts)[:, None] + offsets[None, :]]
    return mask_windows(windows, config.context_steps, gap_length,
                        len(config.sensor_features), config.sensor_features.index(config.target),
                        config.scenario)


class RandomGapBatches:
    """Sample all training years with a reproducible, epoch-dependent generator.

    One random gap length is chosen per batch, allowing exact context lengths
    without padding. Across batches/epochs all configured lengths are sampled.
    """

    def __init__(self, values, config):
        self.values = values
        self.config = config
        self.candidates = {length: valid_gap_starts(values, config.sampling_context_steps, length)
                           for length in config.gap_lengths}
        if any(len(positions) == 0 for positions in self.candidates.values()):
            raise ValueError("Training data has no complete window for a requested gap length.")

    def epoch(self, epoch_number):
        """Draw exactly samples_per_epoch windows, including a final small batch."""
        config = self.config
        rng = np.random.default_rng(config.seed + epoch_number)
        for offset in range(0, config.samples_per_epoch, config.batch_size):
            size = min(config.batch_size, config.samples_per_epoch - offset)
            length = int(rng.choice(config.gap_lengths))
            starts = rng.choice(self.candidates[length], size=size, replace=True)
            yield gather_batch(self.values, starts, length, config)


def evaluation_batches(values, manifest, config):
    """Yield fixed cases grouped by length, along with their manifest rows."""
    for length, rows in manifest.groupby("gap_length", sort=True):
        for offset in range(0, len(rows), config.batch_size):
            selected = rows.iloc[offset:offset + config.batch_size]
            batch = gather_batch(values, selected.gap_start.to_numpy(), int(length), config)
            yield batch, selected
