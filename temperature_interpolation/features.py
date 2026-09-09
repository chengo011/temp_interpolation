from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

"""Leakage-safe sensor selection, cyclic time features, and train-only scaling."""

SAFE_SENSOR_FEATURES = [
    "T (degC)", "p (mbar)", "rh (%)", "wind_x", "wind_y", "max_wind_x", "max_wind_y",
]
TIME_FEATURES = ["day_sin", "day_cos", "year_sin", "year_cos"]
EXCLUDED_FEATURES = {
    "Tpot (K)": "Potential temperature is derived from temperature and pressure.",
    "Tdew (degC)": "Dew point together with humidity can encode temperature.",
    "VPmax (mbar)": "Saturation vapor pressure is a function of temperature.",
    "VPact (mbar)": "Vapor pressure combined with humidity can encode temperature.",
    "VPdef (mbar)": "Vapor pressure deficit contains temperature-dependent information.",
    "sh (g/kg)": "Specific humidity combines with other weather variables to recover temperature.",
    "H2OC (mmol/mol)": "Water concentration may indirectly recover temperature with other inputs.",
    "rho (g/m**3)": "Air density is linked algebraically to pressure and temperature.",
}


def engineer_features(frame: pd.DataFrame, sensor_features: list[str]):
    """Encode circular variables continuously without copying target derivatives."""
    if not set(sensor_features) <= set(SAFE_SENSOR_FEATURES):
        raise ValueError("Unsafe or unknown sensor feature requested.")
    result = frame.copy()
    if any("wind" in feature for feature in sensor_features):
        direction = np.deg2rad(result["wd (deg)"])
        result["wind_x"] = result["wv (m/s)"] * np.cos(direction)
        result["wind_y"] = result["wv (m/s)"] * np.sin(direction)
        result["max_wind_x"] = result["max. wv (m/s)"] * np.cos(direction)
        result["max_wind_y"] = result["max. wv (m/s)"] * np.sin(direction)
    result = result[sensor_features].copy()
    minutes = frame.index.hour * 60 + frame.index.minute
    day_phase = np.asarray(minutes, dtype=float) / (24 * 60)
    # Calendar-year phase explicitly handles leap years instead of drifting
    # by a fixed 365-day period across the eight-year dataset.
    year_days = np.where(frame.index.is_leap_year, 366.0, 365.0)
    year_phase = (np.asarray(frame.index.dayofyear) - 1 + day_phase) / year_days
    result["day_sin"] = np.sin(2 * np.pi * day_phase)
    result["day_cos"] = np.cos(2 * np.pi * day_phase)
    result["year_sin"] = np.sin(2 * np.pi * year_phase)
    result["year_cos"] = np.cos(2 * np.pi * year_phase)
    return result


@dataclass
class Standardizer:
    """Store training statistics, column order, and fitting provenance."""

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    fit_start: str
    fit_end: str

    @classmethod
    def fit(cls, training_features: pd.DataFrame):
        """Estimate statistics on observed training values only, ignoring NaNs."""
        mean = training_features.mean().to_numpy(dtype=np.float64)
        scale = training_features.std(ddof=0).to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Every input feature needs finite training observations.")
        scale[scale < 1e-8] = 1.0
        return cls(list(training_features.columns), mean, scale,
                   str(training_features.index.min()), str(training_features.index.max()))

    def transform(self, features: pd.DataFrame):
        """Apply the frozen column order and statistics without refitting."""
        if list(features.columns) != self.columns:
            raise ValueError("Feature columns do not match the fitted preprocessing.")
        return ((features.to_numpy(dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def inverse_target(self, values, target="T (degC)"):
        """Convert normalized model output back to degrees Celsius."""
        index = self.columns.index(target)
        return np.asarray(values) * self.scale[index] + self.mean[index]

    def save(self, path: Path):
        """Save reusable statistics and the documented feature exclusions."""
        payload = {
            "columns": self.columns, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "fit_start": self.fit_start, "fit_end": self.fit_end,
            "excluded_features": EXCLUDED_FEATURES,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        """Restore preprocessing for inference without accessing training data."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["columns"], np.array(payload["mean"]), np.array(payload["scale"]),
                   payload["fit_start"], payload["fit_end"])
