# ECG Anomaly Lab

A Data Science course project exploring ECG signals and anomaly detection.
It uses the **MIT-BIH Arrhythmia Database 1.0.0**: 48 approximately half-hour,
two-channel recordings from 47 subjects, sampled at 360 Hz, with expert-reviewed
beat and event annotations. This is a learning and research project.

## Start

Extract this starter into a **new folder**, separate from course materials and
existing work. In WSL, open that folder in VS Code and run:

```bash
uv sync --locked
uv run python scripts/verify_setup.py
```

Select `.venv/bin/python` as the notebook kernel and open
`notebooks/00_start.ipynb`. Alternatively, run `uv run jupyter lab`.
Python 3.15.2 and dependencies are managed by uv; initial installation needs internet.
Use a current uv release so it can download the pinned Python version.

## Included

- All 48 raw records: `.hea` headers, `.dat` signals, `.atr` annotations in
  `data/raw/mitdb-1.0.0/`. No filtering, windowing, label mapping, or split applied.
- `data/manifest.json`: source version, file URLs, and SHA-256 checksums.
- `notebooks/EDA.ipynb`: signal/window exploration, preprocessing, and dataset audits.
- `data/record_splits.csv`: reproducible train/validation/test assignments by subject.
- [Code and data guide](docs/CODE_GUIDE.md): function reference, selection rules,
  notebook navigation, and reproduction commands.
- An environment/data integrity check and automated tests.
- `AGENTS.md`

Raw data and the environment are excluded from Git. After cloning a code-only
repository, restore missing data with `uv run python scripts/fetch_data.py`.
This checks existing files and does not overwrite mismatching local data.


Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
Dataset attribution and license: [DATA_NOTICE.md](DATA_NOTICE.md).
Validation status: [SETUP_CHECK.md](SETUP_CHECK.md).
