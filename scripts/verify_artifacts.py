"""Independently audit completed artifacts without changing any fitted models."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_suite(run_name):
    """Recompute metrics and check saved models, plots, and common test positions."""
    import torch
    from temperature_interpolation.config import ExperimentConfig
    from temperature_interpolation.features import Standardizer
    from temperature_interpolation.evaluation import summarize_predictions

    suite_dir = ROOT / "results" / run_name
    completion = json.loads((suite_dir / "complete.json").read_text())
    protocol = json.loads((suite_dir / "protocol.json").read_text())
    combined = pd.read_csv(suite_dir / "all_test_metrics.csv")
    expected_manifest = pd.read_csv(ROOT / "data" / "processed" / "test_gaps.csv")
    expected_keys = {(row.gap_id, offset) for row in expected_manifest.itertuples() for offset in range(row.gap_length)}
    assert completion["experiments"] == len(protocol["experiments"])
    for relative_path, expected_hash in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash
    for split_name, expected_hash in protocol["manifest_sha256"].items():
        assert hashlib.sha256((ROOT / "data" / "processed" / f"{split_name}_gaps.csv").read_bytes()).hexdigest() == expected_hash
    checks = []
    for settings in protocol["experiments"]:
        config = ExperimentConfig(**settings)
        output_dir = ROOT / "models" / run_name / config.name
        saved_config = ExperimentConfig.load(output_dir / "config.json")
        assert saved_config == config
        scaler = Standardizer.load(output_dir / "preprocessing.json")
        assert pd.Timestamp(scaler.fit_end).year == max(config.train_years)
        history = pd.read_csv(output_dir / "history.csv")
        checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
        metadata = json.loads((output_dir / "training_metadata.json").read_text())
        best_row = history.loc[history.validation_masked_mse.idxmin()]
        assert checkpoint["best_epoch"] == int(best_row.epoch) == metadata["best_epoch"]
        assert np.isclose(checkpoint["validation_masked_mse"], best_row.validation_masked_mse)
        predictions = pd.read_csv(output_dir / "test_predictions.csv.gz")
        for method, rows in predictions.groupby("method"):
            assert set(zip(rows.gap_id, rows.offset)) == expected_keys, method
            assert len(rows) == len(expected_keys)
        saved_metrics = pd.read_csv(output_dir / "test_metrics.csv")
        recomputed = summarize_predictions(predictions)
        keys = ["method", "gap_length"]
        pd.testing.assert_frame_equal(saved_metrics.sort_values(keys).reset_index(drop=True),
                                      recomputed.sort_values(keys).reset_index(drop=True), rtol=1e-6, atol=1e-8)
        assert set(saved_metrics.method) == {"Forward fill", "Linear", "Cubic spline", "PCHIP", "BiLSTM"}
        assert (saved_metrics.gap_count == config.test_gaps_per_length).all()
        # Compare evaluation ground truth directly against the clean source,
        # ensuring stored truths were neither normalized nor overwritten.
        cleaned = pd.read_pickle(ROOT / "data" / "processed" / "cleaned.pkl")
        point_truth = cleaned.loc[pd.to_datetime(predictions.timestamp), config.target].to_numpy()
        np.testing.assert_allclose(predictions.true_temperature, point_truth, atol=1e-10)
        plot_paths = list((output_dir / "plots").glob("*.png"))
        assert len(plot_paths) == 3 + len(config.gap_lengths)
        assert all(path.stat().st_size > 10000 for path in plot_paths)
        assert (output_dir / "validation_metrics.csv").exists()
        checks.append({"experiment": config.name, "best_epoch": checkpoint["best_epoch"],
                       "epochs_completed": len(history), "test_points_per_method": len(expected_keys),
                       "verified_plot_count": len(plot_paths), "status": "passed"})
    assert len(combined) == len(protocol["experiments"]) * 5 * len(protocol["experiments"][0]["gap_lengths"])
    result = {"status": "passed", "experiments": checks,
              "checks": ["source and manifest hashes", "best validation checkpoint", "train-only scaler provenance",
                         "identical test cases", "independent metric recomputation", "unaltered Celsius truth",
                         "complete plots", "complete experiment matrix"]}
    (suite_dir / "artifact_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_experiment")
    verify_suite(parser.parse_args().run_name)
