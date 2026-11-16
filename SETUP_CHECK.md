# Setup verification

Verified on 2026-11-14 using Linux x86_64, Python 3.15.2, and uv 0.13.6:

- `uv sync --locked` created a fresh Python 3.15.2 environment successfully.
  The lockfile was regenerated for Python 3.15; package versions were preserved.
- `uv pip check` confirmed compatibility of all 115 installed packages.
- All eight declared runtime dependencies imported successfully.
- NumPy, pandas, SciPy, and scikit-learn completed small numerical checks;
  matplotlib rendered a PNG successfully.
- All 48 records and 144 raw files passed size and SHA-256 checks.
- WFDB read every header, annotation file, and the first signal sample;
  annotation positions were within the corresponding recording bounds.
- The starter notebook passed format validation and its code cell executed
  successfully as Python. The JupyterLab command reported version 4.6.4.

Jupyter kernel execution was attempted, but this environment denied local
network operations (`Operation not permitted`), causing the kernel to exit.
Interactive VS Code/Jupyter operation therefore remains unverified here; select
this project's `.venv/bin/python` locally. These checks cover the current starter
dependencies, not packages added later.

Python release: [Python 3.15.2](https://www.python.org/downloads/release/python-3152/).
