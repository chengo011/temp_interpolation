
from pathlib import Path
import sys
import numpy as np
import pandas as pd

"""Render a saved-model example from validation data for visual inspection."""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_example(scenario="A"):
    from temperature_interpolation.baselines import interpolate_temperature
    from temperature_interpolation.visualization import COLORS, plt, save_figure

    directory = ROOT / "examples"
    reference = pd.read_csv(directory / "reference_observations.csv")
    result = pd.read_csv(directory / f"reconstructed_scenario_{scenario}.csv")
    missing = result.is_reconstructed.to_numpy(dtype=bool)
    gap_start = np.flatnonzero(missing)[0]
    minutes = (np.arange(len(result)) - gap_start) * 10
    truth = reference["T (degC)"].to_numpy()
    damaged = result["T (degC)"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(minutes, truth, color="#263238", linestyle="--", label="True temperature")
    axis.scatter(minutes[~missing], truth[~missing], s=12, color="#263238", label="Observed temperature")
    axis.axvspan(-5, missing.sum() * 10 - 5, color="#e9c46a", alpha=0.25, label="Artificial gap")
    for method in ["Linear", "Cubic spline"]:
        estimates = interpolate_temperature(damaged, missing, method)
        axis.plot(minutes[missing], estimates, marker="o", markersize=3, color=COLORS[method], label=method)
    axis.plot(minutes[missing], result.reconstructed_temperature[missing], marker="o", markersize=3,
              color=COLORS["BiLSTM"], label="BiLSTM")
    axis.set(xlabel="Minutes relative to gap start", ylabel="Temperature (°C)",
             title=f"Saved-model application, scenario {scenario} | validation example (2015)",
             xlim=(-180, missing.sum() * 10 + 180))
    axis.legend(ncol=2, fontsize=9)
    axis.grid(alpha=0.15)
    save_figure(figure, directory / f"reconstruction_scenario_{scenario}")


if __name__ == "__main__":
    plot_example()
