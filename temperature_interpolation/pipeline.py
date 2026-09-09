from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import numpy as np
import pandas as pd

from .cleaning import prepare_data, split_frame
from .config import ExperimentConfig, experiment_matrix
from .download import download_dataset
from .features import SAFE_SENSOR_FEATURES, Standardizer, engineer_features
from .gaps import make_manifest

"""Ordered project execution: prepare, validate, train all, then test once."""

def write_json(path, payload):
    """Write human-readable metadata with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_project(root: Path):
    """Prepare one shared experimental population before fitting any model."""
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)
    configs = experiment_matrix()
    for config in configs:
        path = config_dir / f"{config.name}.json"
        if not path.exists():
            config.save(path)
    # Reading the saved files lets users configure later reruns centrally.
    configs = [ExperimentConfig.load(config_dir / f"{config.name}.json") for config in configs]
    base = configs[0]
    for config in configs:
        for field in ["train_years", "validation_years", "test_years", "gap_lengths", "seed",
                      "sampling_context_steps", "validation_gaps_per_length", "test_gaps_per_length"]:
            if getattr(config, field) != getattr(base, field):
                raise ValueError(f"All suite configurations must share {field} for matched comparisons.")
    frame, audit = prepare_data(download_dataset(root), root)
    splits = split_frame(frame, base)
    processed_dir = root / "data" / "processed"
    summary = {}
    for name, split in splits.items():
        features = engineer_features(split, SAFE_SENSOR_FEATURES)
        summary[name] = {
            "rows": len(split), "start": str(split.index.min()), "end": str(split.index.max()),
            "complete_safe_sensor_rows": int(np.isfinite(features.to_numpy()).all(axis=1).sum()),
        }
        if name == "train":
            continue
        count = base.validation_gaps_per_length if name == "validation" else base.test_gaps_per_length
        seed = base.seed + (10000 if name == "validation" else 20000)
        manifest = make_manifest(features, base.gap_lengths, base.sampling_context_steps, count, seed)
        path = processed_dir / f"{name}_gaps.csv"
        if path.exists():
            existing = pd.read_csv(path)
            pd.testing.assert_frame_equal(existing, manifest, check_dtype=False)
        else:
            manifest.to_csv(path, index=False)
    summary["excluded_outside_configured_years"] = len(frame) - sum(len(split) for split in splits.values())
    write_json(processed_dir / "split_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return configs, splits


def prepare_experiment(splits, config):
    """Fit train-only scaling and preserve identical eligibility across runs.

    Even the temperature-only run uses the shared valid-sensor population.
    This isolates the effect of sensor availability from sample selection.
    Setting invalid rows to NaN does not impute or alter any observed values.
    """
    features = {name: engineer_features(split, config.sensor_features) for name, split in splits.items()}
    standardizer = Standardizer.fit(features["train"])
    normalized = {}
    for name, split in splits.items():
        normalized[name] = standardizer.transform(features[name])
        common_features = engineer_features(split, SAFE_SENSOR_FEATURES)
        bad_rows = ~np.isfinite(common_features.to_numpy()).all(axis=1)
        normalized[name][bad_rows] = np.nan
    return features, normalized, standardizer


def run_suite(root: Path, run_name: str):
    """Execute the frozen suite and persist each stage for safe continuation."""
    from .evaluation import evaluate_baselines, evaluate_model, save_evaluation
    from .training import load_trained_model, train_model, set_reproducibility
    from .visualization import plot_run, plot_experiment_comparisons
    from .reporting import generate_report

    configs, splits = prepare_project(root)
    suite_dir = root / "results" / run_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    model_dir = root / "models" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    validation_manifest = pd.read_csv(root / "data" / "processed" / "validation_gaps.csv")
    test_manifest = pd.read_csv(root / "data" / "processed" / "test_gaps.csv")
    manifests = {name: hashlib.sha256((root / "data" / "processed" / f"{name}_gaps.csv").read_bytes()).hexdigest()
                 for name in ["validation", "test"]}
    protocol_path = suite_dir / "protocol.json"
    protocol = {
        "experiments": [config.__dict__ for config in configs], "manifest_sha256": manifests,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((root / "temperature_interpolation").glob("*.py"))},
        "data_sha256": json.loads((root / "data" / "raw" / "provenance.json").read_text()),
    }
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Code, data, or settings changed. Use a new run name to avoid mixing results.")
    else:
        write_json(protocol_path, protocol)
        write_json(suite_dir / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        dependencies = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (suite_dir / "requirements.lock.txt").write_text(dependencies, encoding="utf-8")

    # Baseline validation precedes DL training, as required by the project guide.
    for context in sorted({config.context_steps for config in configs}):
        baseline_dir = suite_dir / f"validation_baselines_context_{context // 6}h"
        baseline_dir.mkdir(exist_ok=True)
        baseline_config = ExperimentConfig(context_steps=context)
        baseline_config.target = configs[0].target
        if not (baseline_dir / "validation_metrics.csv").exists():
            validation_features = engineer_features(splits["validation"], SAFE_SENSOR_FEATURES)
            predictions = evaluate_baselines(validation_features, validation_manifest, baseline_config)
            save_evaluation(predictions, baseline_dir, "validation")

    # Finish ALL validation-selected models before reading any test predictions.
    # Test features are transformed using fixed statistics, but are never passed
    # to the optimizer, scheduler, checkpoint selection, or architecture choice.
    for config in configs:
        output_dir = model_dir / config.name
        if (output_dir / "training_metadata.json").exists():
            print(f"Reusing completed training: {config.name}", flush=True)
            continue
        if (output_dir / "best_model.pt").exists():
            raise RuntimeError("An interrupted training run exists. Use a new run name to retrain cleanly.")
        features, normalized, standardizer = prepare_experiment(splits, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        standardizer.save(output_dir / "preprocessing.json")
        model = train_model(normalized["train"], normalized["validation"], validation_manifest, config, output_dir)
        predictions = pd.concat([
            evaluate_baselines(features["validation"], validation_manifest, config),
            evaluate_model(model, normalized["validation"], features["validation"], validation_manifest, config, standardizer),
        ], ignore_index=True)
        save_evaluation(predictions, output_dir, "validation")
    write_json(suite_dir / "training_complete.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "all_models_selected_on": "validation only",
    })

    combined = []
    for config in configs:
        output_dir = model_dir / config.name
        set_reproducibility(config.seed, config.threads)
        features, normalized, _ = prepare_experiment(splits, config)
        standardizer = Standardizer.load(output_dir / "preprocessing.json")
        # Explicitly use the SAVED scaler during final evaluation.
        test_values = standardizer.transform(features["test"])
        model = load_trained_model(output_dir, config)
        predictions = pd.concat([
            evaluate_baselines(features["test"], test_manifest, config),
            evaluate_model(model, test_values, features["test"], test_manifest, config, standardizer),
        ], ignore_index=True)
        metrics = save_evaluation(predictions, output_dir, "test")
        plot_run(features["test"], test_manifest, predictions, metrics, config, output_dir)
        metrics["experiment"] = config.name
        metrics["scenario"] = config.scenario
        metrics["context_hours"] = config.context_steps / 6
        combined.append(metrics)
        print(f"Test evaluation completed: {config.name}", flush=True)
    combined = pd.concat(combined, ignore_index=True)
    combined.to_csv(suite_dir / "all_test_metrics.csv", index=False)
    plot_experiment_comparisons(combined, suite_dir)
    generate_report(root, suite_dir, model_dir, combined)
    write_json(suite_dir / "complete.json", {"finished_utc": datetime.now(timezone.utc).isoformat(), "experiments": len(configs)})
    print(f"Complete results: {suite_dir}", flush=True)
