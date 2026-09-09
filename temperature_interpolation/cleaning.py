
from pathlib import Path
import json
import numpy as np
import pandas as pd

"""Audit timestamps and invalidate implausible observations without imputation."""

# Wide physical bounds catch clear errors rather than removing unusual weather.
# No threshold is fitted to validation or test data.
PHYSICAL_BOUNDS = {
    "T (degC)": (-90, 60), "p (mbar)": (300, 1100),
    "rh (%)": (0, 100), "wv (m/s)": (0, 100),
    "max. wv (m/s)": (0, 150), "wd (deg)": (0, 360),
    "Tpot (K)": (150, 400), "Tdew (degC)": (-100, 60),
    "VPmax (mbar)": (0, 250), "VPact (mbar)": (0, 250),
    "VPdef (mbar)": (0, 250), "sh (g/kg)": (0, 100),
    "H2OC (mmol/mol)": (0, 200), "rho (g/m**3)": (500, 2000),
}


def clean_frame(raw: pd.DataFrame):
    """Return a regular ten-minute grid and an explicit cleaning audit.

    Duplicate timestamps retain the first record in original file order. All
    invalid cells become NaN. Missing timestamps become empty rows, so later
    window selection cannot silently bridge an actual recording outage.
    No interpolation is performed here: it could leak hidden target values.
    """
    frame = raw.copy(deep=True)
    timestamps = pd.to_datetime(frame.pop("Date Time"), format="%d.%m.%Y %H:%M:%S", errors="coerce")
    audit = {"raw_rows": len(frame), "invalid_timestamps": int(timestamps.isna().sum())}
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    frame = frame.loc[frame.index.notna()].sort_index(kind="stable")
    audit["duplicate_timestamps"] = int(frame.index.duplicated().sum())
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    differences = frame.index.to_series().diff().dropna().dt.total_seconds()
    audit["interval_counts_seconds"] = {str(int(k)): int(v) for k, v in differences.value_counts().items()}
    # The source uses naive station timestamps. Preserve these labels; do not
    # invent a time-zone conversion or collapse daylight-saving transitions.
    on_grid = (frame.index.minute % 10 == 0) & (frame.index.second == 0)
    audit["off_grid_rows"] = int((~on_grid).sum())
    frame = frame.loc[on_grid].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    audit["missing_before_physical_checks"] = frame.isna().sum().astype(int).to_dict()
    audit["invalid_physical_values"] = {}
    for column, (lower, upper) in PHYSICAL_BOUNDS.items():
        if column in frame:
            invalid = frame[column].notna() & ~frame[column].between(lower, upper)
            audit["invalid_physical_values"][column] = int(invalid.sum())
            frame.loc[invalid, column] = np.nan
    regular_index = pd.date_range(frame.index.min(), frame.index.max(), freq="10min", name="timestamp")
    audit["inserted_missing_timestamps"] = len(regular_index) - len(frame)
    frame = frame.reindex(regular_index)
    audit["cleaned_rows"] = len(frame)
    audit["unknown_temperature_rows"] = int(frame["T (degC)"].isna().sum())
    audit["first_timestamp"] = str(frame.index.min())
    audit["last_timestamp"] = str(frame.index.max())
    return frame, audit


def prepare_data(csv_path: Path, root: Path):
    """Persist cleaned data and its audit in the project directory."""
    frame, audit = clean_frame(pd.read_csv(csv_path))
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(processed_dir / "cleaned.pkl")
    (processed_dir / "cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return frame, audit


def split_frame(frame, config):
    """Split BEFORE generating windows, so no window crosses split boundaries."""
    config.validate()
    splits = {
        "train": frame.loc[frame.index.year.isin(config.train_years)].copy(),
        "validation": frame.loc[frame.index.year.isin(config.validation_years)].copy(),
        "test": frame.loc[frame.index.year.isin(config.test_years)].copy(),
    }
    for split in splits.values():
        if split.empty:
            raise ValueError("A requested chronological split is empty.")
        if not split.index.is_monotonic_increasing or not split.index.is_unique:
            raise ValueError("Timestamps must be sorted and unique.")
        if not (split.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all():
            raise ValueError("Each split must retain an uninterrupted ten-minute index.")
    assert splits["train"].index.max() < splits["validation"].index.min()
    assert splits["validation"].index.max() < splits["test"].index.min()
    return splits
