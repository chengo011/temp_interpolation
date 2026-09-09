
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

"""Download the original ten-minute data without resampling it."""

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"


def download_dataset(root: Path) -> Path:
    """Download atomically, extract only the known CSV, and record provenance.

    Explicit member selection avoids extracting unexpected archive paths.
    Existing downloads are reused, and SHA-256 hashes make the actual data
    version auditable even if the upstream resource later changes.
    """
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "jena_climate_2009_2016.csv.zip"
    csv_path = raw_dir / "jena_climate_2009_2016.csv"
    if not archive.exists():
        temporary_path = archive.with_suffix(".partial")
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            with temporary_path.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
        temporary_path.replace(archive)
    if not csv_path.exists():
        with zipfile.ZipFile(archive) as zipped:
            csv_path.write_bytes(zipped.read(csv_path.name))
    metadata = {
        "url": DATA_URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (raw_dir / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path
