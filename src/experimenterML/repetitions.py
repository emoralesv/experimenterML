from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from .logging_utils import append_row, ensure_dir, write_json


def _grid(param_grid: Dict[str, Iterable]) -> List[Dict]:
    keys = sorted(param_grid.keys())
    vals = [list(param_grid[k]) for k in keys]
    return [dict(zip(keys, v)) for v in product(*vals)]


def short_uid(d: Dict, keys: List[str]) -> str:
    """Compute 8-char uid from selected keys using sha256.

    Args:
        d: Parameter dict.
        keys: Keys to include in the hash.

    Returns:
        8-character uid hex string.
    """
    picked = {k: d.get(k) for k in sorted(keys)}
    payload = json.dumps(picked, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass
class Repetitions:
    """Run combinatorial experiments with hashing and logging.

    Attributes:
        train_fn: Callable that receives (params, run_id) and returns metrics dict.
        param_grid: Dict of lists for grid.
        hash_keys: Keys to include in uid generation.
        out_dir: Output directory for CSV/JSON results.
        csv_name: File name for summary CSV.
    """

    train_fn: Callable[[Dict, str], Dict]
    param_grid: Dict[str, Iterable]
    hash_keys: List[str]
    out_dir: str = "results"
    csv_name: str = "experiments.csv"

    def run_all(self, parallel: bool = False) -> None:
        del parallel  # sequential by default for determinism
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(self.out_dir) / self.csv_name
        existing = set()
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add(row.get("uid", ""))
        for params in _grid(self.param_grid):
            uid = short_uid(params, self.hash_keys)
            if uid in existing:
                continue
            metrics = self.train_fn(params, uid)
            row = {**params, **metrics, "uid": uid}
            append_row(csv_path, row)
            write_json(Path(self.out_dir) / "json" / f"{uid}.json", {"params": params, "metrics": metrics})

