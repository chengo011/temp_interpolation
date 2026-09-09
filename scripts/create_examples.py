
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

"""Prepare real validation-year examples and run the completed saved models."""

def create_examples(run_name, reconstruct):
    from temperature_interpolation.config import ExperimentConfig
    from temperature_interpolation.inference import interpolate_csv

    output_dir = ROOT / "examples"
    output_dir.mkdir(exist_ok=True)
    frame = pd.read_pickle(ROOT / "data" / "processed" / "cleaned.pkl")
    config = ExperimentConfig.load(ROOT / "configs" / "A_multivariate_context_12h.json")
    validation = frame.loc[frame.index.year.isin(config.validation_years)]
    manifest = pd.read_csv(ROOT / "data" / "processed" / "validation_gaps.csv")
    row = manifest.loc[manifest.gap_length == 12].iloc[0]
    start, length, context = int(row.gap_start), int(row.gap_length), config.context_steps
    original = validation.iloc[start - context:start + length + context].copy()
    original.to_csv(output_dir / "reference_observations.csv", index_label="timestamp")
    for scenario in ["A", "B"]:
        damaged = original.copy(deep=True)
        if scenario == "A":
            damaged.iloc[context:context + length, damaged.columns.get_loc(config.target)] = np.nan
        else:
            damaged.iloc[context:context + length, :] = np.nan
        damaged.insert(0, "Date Time", damaged.index.strftime("%d.%m.%Y %H:%M:%S"))
        input_path = output_dir / f"input_scenario_{scenario}.csv"
        damaged.to_csv(input_path, index=False)
        if reconstruct:
            model_dir = ROOT / "models" / run_name / f"{scenario}_multivariate_context_12h"
            result = interpolate_csv(input_path, output_dir / f"reconstructed_scenario_{scenario}.csv", model_dir)
            assert result.is_reconstructed.sum() == length
            assert result.reconstructed_temperature.notna().all()
            observed = ~result.is_reconstructed.to_numpy()
            np.testing.assert_array_equal(result.reconstructed_temperature.to_numpy()[observed],
                                          original[config.target].to_numpy()[observed])
            print(f"Verified saved-model inference for scenario {scenario}: {length} reconstructed values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_experiment")
    parser.add_argument("--reconstruct", action="store_true")
    arguments = parser.parse_args()
    create_examples(arguments.run_name, arguments.reconstruct)
