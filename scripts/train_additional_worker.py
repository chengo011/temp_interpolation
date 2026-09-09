"""Train one predeclared late experiment in an independent Python process.

Only frozen protocol settings are accepted. This worker does not evaluate the
test set, change hyperparameters, or introduce an additional model variant.
The main pipeline will reuse the completed model when it reaches this run.
"""

from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def train_one(run_name, experiment_name):
    """Fit the named experiment with exactly the saved suite configuration."""
    from temperature_interpolation.cleaning import split_frame
    from temperature_interpolation.config import ExperimentConfig
    from temperature_interpolation.evaluation import evaluate_baselines, evaluate_model, save_evaluation
    from temperature_interpolation.pipeline import prepare_experiment
    from temperature_interpolation.training import train_model

    protocol = json.loads((ROOT / "results" / run_name / "protocol.json").read_text())
    settings = next(item for item in protocol["experiments"] if item["name"] == experiment_name)
    config = ExperimentConfig(**settings)
    config.validate()
    output_dir = ROOT / "models" / run_name / config.name
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_pickle(ROOT / "data" / "processed" / "cleaned.pkl")
    splits = split_frame(frame, config)
    features, normalized, standardizer = prepare_experiment(splits, config)
    standardizer.save(output_dir / "preprocessing.json")
    manifest = pd.read_csv(ROOT / "data" / "processed" / "validation_gaps.csv")
    model = train_model(normalized["train"], normalized["validation"], manifest, config, output_dir)
    predictions = pd.concat([
        evaluate_baselines(features["validation"], manifest, config),
        evaluate_model(model, normalized["validation"], features["validation"], manifest, config, standardizer),
    ], ignore_index=True)
    save_evaluation(predictions, output_dir, "validation")
    print(f"Independent training completed: {config.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_experiment")
    parser.add_argument("--experiment", required=True)
    arguments = parser.parse_args()
    train_one(arguments.run_name, arguments.experiment)
