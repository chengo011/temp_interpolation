from pathlib import Path
import os

# Keep font caches and all generated artifacts inside the project directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Forward fill": "#9299a3", "Linear": "#2471a3", "Cubic spline": "#dc7633",
          "PCHIP": "#7d3c98", "BiLSTM": "#148f77"}


def save_figure(figure, path):
    """Save both a preview PNG and a vector PDF for reuse in a report."""
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_run(raw_features, manifest, predictions, metrics, config, output_dir):
    """Create the four required plot types for a completed experiment."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for method, rows in metrics.groupby("method", sort=False):
        rows = rows.sort_values("gap_minutes")
        axis.plot(rows.gap_minutes, rows.mae_celsius, marker="o", label=method, color=COLORS[method])
    axis.set(xlabel="Gap duration (minutes)", ylabel="Test MAE (°C)", title=config.name)
    axis.set_xticks([length * 10 for length in config.gap_lengths])
    axis.grid(alpha=0.2)
    axis.legend()
    save_figure(figure, plot_dir / "mae_by_gap_length")

    # Macro-average weights each gap length equally. A pooled point average
    # would give long gaps much greater weight; explicitly label this choice.
    average = metrics.groupby("method").mae_celsius.mean().sort_values()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(average.index, average.values, color=[COLORS[item] for item in average.index])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(ylabel="Test MAE (°C), equal weight per gap length", title="Average reconstruction error")
    axis.set_ylim(0, average.max() * 1.2)
    save_figure(figure, plot_dir / "model_comparison")

    history = pd.read_csv(output_dir / "history.csv")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history.epoch, history.train_masked_mse, label="Training (random gaps)")
    axis.plot(history.epoch, history.validation_masked_mse, label="Validation (fixed gaps)")
    axis.set(xlabel="Epoch", ylabel="Masked MSE (normalized temperature)", title="Training history")
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, plot_dir / "training_history")

    # Always use the first saved gap for each length: examples are not chosen
    # after viewing errors and cannot cherry-pick successful reconstructions.
    for length in config.gap_lengths:
        row = manifest.loc[manifest.gap_length == length].iloc[0]
        start = int(row.gap_start)
        context = min(config.context_steps, 18)
        section = raw_features.iloc[start - context:start + length + context]
        relative_minutes = np.arange(-context, length + context) * 10
        truth = section[config.target].to_numpy()
        visible = np.ones(len(section), dtype=bool)
        visible[context:context + length] = False
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(relative_minutes, truth, color="#202b38", linestyle="--", label="True temperature")
        axis.scatter(relative_minutes[visible], truth[visible], s=12, color="#202b38", label="Observed temperature")
        axis.axvspan(-5, length * 10 - 5, color="#e9c46a", alpha=0.25, label="Artificial gap")
        selected = predictions.loc[predictions.gap_id == row.gap_id]
        for method in ["Linear", "Cubic spline", "BiLSTM"]:
            values = selected.loc[selected.method == method].sort_values("offset")
            axis.plot(values.offset * 10, values.predicted_temperature, marker="o", markersize=3,
                      label=method, color=COLORS[method])
        axis.set(xlabel="Minutes relative to gap start", ylabel="Temperature (°C)",
                 title=f"{length * 10}-minute gap | {row.timestamp} | scenario {config.scenario}")
        axis.legend(ncol=2, fontsize=8)
        save_figure(figure, plot_dir / f"reconstruction_{length:02d}_steps")


def plot_experiment_comparisons(combined, output_dir):
    """Show the feature, outage, and context comparisons on common test cases."""
    selected = combined.loc[(combined.method == "BiLSTM") & (combined.context_hours == 12)]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for experiment, rows in selected.groupby("experiment"):
        label = "Temperature + time" if "temperature_only" in experiment else f"Multivariate, scenario {rows.scenario.iloc[0]}"
        axis.plot(rows.gap_minutes, rows.mae_celsius, marker="o", label=label)
    axis.set(xlabel="Gap duration (minutes)", ylabel="Test MAE (°C)", title="Effect of available sensor information")
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, output_dir / "sensor_availability_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for scenario, axis in zip(["A", "B"], axes, strict=True):
        selected = combined.loc[(combined.method == "BiLSTM") & (combined.scenario == scenario)
                                & ~combined.experiment.str.contains("temperature_only")]
        for hours, rows in selected.groupby("context_hours"):
            axis.plot(rows.gap_minutes, rows.mae_celsius, marker="o", label=f"{hours:g} h per side")
        axis.set(xlabel="Gap duration (minutes)", title=f"Scenario {scenario}")
        axis.legend()
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Test MAE (°C)")
    save_figure(figure, output_dir / "context_length_comparison")
