
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

"""Classical methods that only receive observed temperature values."""


METHODS = ["Forward fill", "Linear", "Cubic spline", "PCHIP"]


def interpolate_temperature(damaged, missing_mask, method):
    """Reconstruct a bounded gap without ever using its hidden original values.

    CubicSpline uses the SciPy not-a-knot boundary condition; PCHIP preserves
    local shape. All observed points in the configured context are supplied.
    Predictions are not clipped, so spline overshoot remains visible in scores.
    """
    damaged = np.asarray(damaged, dtype=float)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    positions = np.arange(len(damaged))
    known = ~missing_mask & np.isfinite(damaged)
    wanted = positions[missing_mask]
    if not len(wanted):
        return np.empty(0)
    if not known.any() or wanted.min() <= positions[known].min() or wanted.max() >= positions[known].max():
        raise ValueError("Interpolation requires observed context on both sides.")
    known_positions = positions[known]
    known_values = damaged[known]
    if method == "Linear":
        return np.interp(wanted, known_positions, known_values)
    if method == "Cubic spline":
        return CubicSpline(known_positions, known_values, extrapolate=False)(wanted)
    if method == "PCHIP":
        return PchipInterpolator(known_positions, known_values, extrapolate=False)(wanted)
    if method == "Forward fill":
        previous = np.searchsorted(known_positions, wanted, side="right") - 1
        return known_values[previous]
    raise ValueError(f"Unknown interpolation method: {method}")
