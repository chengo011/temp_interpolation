
import numpy as np
import pandas as pd
import torch

from .baselines import METHODS, interpolate_temperature
from .gaps import evaluation_batches

"""Matched-gap evaluation in Celsius with auditable per-point predictions."""


def summarize_predictions(predictions):
    """Compute MAE and RMSE from original-unit errors for each gap length."""
    records = []
    for (method, length), group in predictions.groupby(["method", "gap_length"], sort=False):
        error = group.predicted_temperature.to_numpy() - group.true_temperature.to_numpy()
        records.append({
            "method": method, "gap_length": int(length), "gap_minutes": int(length) * 10,
            "mae_celsius": float(np.abs(error).mean()),
            "rmse_celsius": float(np.sqrt(np.mean(error ** 2))),
            "gap_count": int(group.gap_id.nunique()), "point_count": len(group),
        })
    return pd.DataFrame(records)


def evaluate_baselines(raw_features, manifest, config):
    """Physically remove hidden target values before calling classical methods."""
    records = []
    temperature = raw_features[config.target].to_numpy()
    context = config.context_steps
    for row in manifest.itertuples(index=False):
        start, length = row.gap_start, row.gap_length
        original = temperature[start - context:start + length + context].copy()
        missing = np.zeros(len(original), dtype=bool)
        missing[context:context + length] = True
        damaged = original.copy()
        damaged[missing] = np.nan
        truth = original[missing].copy()
        for method in METHODS:
            predicted = interpolate_temperature(damaged, missing, method)
            for offset, (actual, estimate) in enumerate(zip(truth, predicted, strict=True)):
                records.append({
                    "gap_id": row.gap_id, "gap_length": length, "offset": offset,
                    "timestamp": str(raw_features.index[start + offset]), "method": method,
                    "true_temperature": float(actual), "predicted_temperature": float(estimate),
                })
    return pd.DataFrame(records)


@torch.inference_mode()
def evaluate_model(model, normalized_values, raw_features, manifest, config, standardizer):
    """Score only masked points after reversing the training normalization."""
    model.eval()
    records = []
    for (inputs, _, _), rows in evaluation_batches(normalized_values, manifest, config):
        predictions = standardizer.inverse_target(model(inputs).cpu().numpy(), config.target)
        for batch_index, row in enumerate(rows.itertuples(index=False)):
            for offset in range(row.gap_length):
                position = row.gap_start + offset
                records.append({
                    "gap_id": row.gap_id, "gap_length": row.gap_length, "offset": offset,
                    "timestamp": str(raw_features.index[position]), "method": "BiLSTM",
                    "true_temperature": float(raw_features.iloc[position][config.target]),
                    "predicted_temperature": float(predictions[batch_index, config.context_steps + offset]),
                })
    return pd.DataFrame(records)


def save_evaluation(predictions, output_dir, split_name):
    """Persist predictions, long-form metrics, and a compact MAE comparison."""
    if not np.isfinite(predictions[["true_temperature", "predicted_temperature"]].to_numpy()).all():
        raise ValueError("Evaluation contains non-finite temperatures.")
    # Every method must be evaluated on identical gap IDs AND point offsets.
    expected = None
    for _, rows in predictions.groupby("method"):
        keys = set(zip(rows.gap_id, rows.offset))
        if len(keys) != len(rows):
            raise ValueError("Duplicate evaluation points detected.")
        if expected is None:
            expected = keys
        elif expected != keys:
            raise ValueError("Methods were evaluated on different gap positions.")
    metrics = summarize_predictions(predictions)
    predictions.to_csv(output_dir / f"{split_name}_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / f"{split_name}_metrics.csv", index=False)
    metrics.pivot(index="method", columns="gap_minutes", values="mae_celsius").to_csv(
        output_dir / f"{split_name}_mae_table.csv"
    )
    return metrics
