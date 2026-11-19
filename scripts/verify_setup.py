"""Check dependencies, raw-file integrity, and WFDB readability without analysis."""

import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check the pinned environment, 144 raw files, and WFDB record readability.

    Returns:
        None. Prints check results and raises on an environment or data mismatch.
    """
    expected_python = (ROOT / ".python-version").read_text().strip()
    if sys.version.split()[0] != expected_python:
        raise RuntimeError(f"Use Python {expected_python} through uv.")
    print(f"Python {sys.version.split()[0]}")
    for package, module in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                            "matplotlib": "matplotlib", "scikit-learn": "sklearn",
                            "wfdb": "wfdb", "jupyterlab": "jupyterlab", "ipykernel": "ipykernel"}.items():
        importlib.import_module(module)
        print(f"OK {package} {version(package)}")

    import numpy as np
    import wfdb

    manifest_path = ROOT / "data/manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run: uv run python scripts/fetch_data.py")
    manifest = json.loads(manifest_path.read_text())
    records_bytes = (ROOT / "data/RECORDS").read_bytes()
    records = records_bytes.decode().splitlines()
    expected_names = {f"{record}.{ext}" for record in records for ext in ("hea", "dat", "atr")}
    if len(records) != 48 or len(set(records)) != 48:
        raise ValueError("Expected 48 unique records.")
    if manifest["version"] != "1.0.0" or manifest["record_count"] != 48:
        raise ValueError("Unexpected dataset version or record count.")
    if manifest["records_file_sha256"] != hashlib.sha256(records_bytes).hexdigest():
        raise ValueError("RECORDS checksum mismatch.")
    if len(manifest["files"]) != 144 or {f["file"] for f in manifest["files"]} != expected_names:
        raise ValueError("The manifest must cover 144 files: three files per record.")
    raw_dir = ROOT / "data/raw/mitdb-1.0.0"
    for entry in manifest["files"]:
        path = raw_dir / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}. Run scripts/fetch_data.py.")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum or size mismatch: {path.name}")
    for record_id in records:
        path = str(raw_dir / record_id)
        header = wfdb.rdheader(path)
        annotation = wfdb.rdann(path, "atr")
        sample = wfdb.rdrecord(path, sampto=1)
        if header.fs != 360 or header.n_sig != 2 or sample.p_signal.shape != (1, 2):
            raise ValueError(f"Unexpected record structure: {record_id}")
        if len(annotation.sample) == 0 or len(annotation.sample) != len(annotation.symbol):
            raise ValueError(f"Invalid annotations: {record_id}")
        if not np.all((annotation.sample >= 0) & (annotation.sample < header.sig_len)):
            raise ValueError(f"Annotation outside recording: {record_id}")
    print("OK: 48 records, 144 matching raw files, readable WFDB signals and annotations.")
    print("Setup checks passed. Notebook UI/kernel startup is a separate local check.")


if __name__ == "__main__":
    main()
