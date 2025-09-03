from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_row(csv_path: Path, row: Dict) -> None:
    ensure_dir(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: Dict) -> None:
    import json

    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

