"""Restore the pinned raw dataset; never overwrite mismatching local files."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"
RAW = ROOT / "data/raw/mitdb-1.0.0"
MANIFEST = ROOT / "data/manifest.json"


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    records = (ROOT / "data/RECORDS").read_text().splitlines()
    if len(records) != 48 or len(set(records)) != 48 or not all(r.isdigit() for r in records):
        raise ValueError("Expected 48 unique record IDs in data/RECORDS.")
    names = [f"{record}.{ext}" for record in records for ext in ("hea", "dat", "atr")]
    existing_manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    expected = {f["file"]: f for f in existing_manifest["files"]} if existing_manifest else {}
    if existing_manifest and (set(expected) != set(names) or existing_manifest["version"] != "1.0.0"):
        raise ValueError("The saved manifest does not match MIT-BIH 1.0.0 record coverage.")
    RAW.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> dict:
        path = RAW / name
        source = BASE_URL + name
        if path.exists():
            raw = path.read_bytes()
            if name in expected and checksum(raw) != expected[name]["sha256"]:
                raise ValueError(f"Local checksum mismatch: {path}. File preserved; inspect it manually.")
        else:
            with urlopen(source, timeout=60) as response:
                raw = response.read()
            if name in expected and checksum(raw) != expected[name]["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {name}. File was not installed.")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(raw)
            temporary.replace(path)
        return {"file": name, "url": source, "bytes": len(raw), "sha256": checksum(raw)}

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, name): name for name in names}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f"{futures[future]}: {exc}")
                print(f"Failed: {failures[-1]}", flush=True)
            if completed % 12 == 0 or completed == len(names):
                print(f"Checked {completed}/{len(names)} files; successful: {len(results)}.", flush=True)
    if failures:
        raise RuntimeError("Some files could not be restored. Existing files were preserved. "
                           "After resolving the errors, rerun the command.\n" + "\n".join(failures))
    if not existing_manifest:
        manifest = {
            "dataset": "MIT-BIH Arrhythmia Database", "version": "1.0.0",
            "source": "https://physionet.org/content/mitdb/1.0.0/",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records), "modifications": "none",
            "records_file_sha256": checksum((ROOT / "data/RECORDS").read_bytes()),
            "files": sorted(results, key=lambda row: row["file"]),
        }
        temporary = MANIFEST.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(MANIFEST)
    print("Raw dataset is ready. No preprocessing or split was applied.")


if __name__ == "__main__":
    main()
