from __future__ import annotations

from pathlib import Path

from experimenterML.repetitions import Repetitions


def train_fn(params: dict, run_id: str) -> dict:  # pragma: no cover - trivial logic
    # Produce deterministic metrics from params
    score = float(params["lr"]) * 1000 + float(params["bs"]) + int(params["seed"]) * 0.1
    return {"accuracy": min(1.0, score / 2000.0), "f1": min(1.0, score / 2500.0), "loss": 1.0 / (score + 1)}


def test_repetitions_runs_and_resumes(tmp_path: Path) -> None:
    grid = {"lr": [1e-3, 1e-4], "bs": [32], "seed": [0, 1]}
    rep = Repetitions(train_fn=train_fn, param_grid=grid, hash_keys=["lr", "bs", "seed"], out_dir=tmp_path.as_posix())
    rep.run_all()
    csv_path = tmp_path / "experiments.csv"
    assert csv_path.exists()
    n_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    # header + 4 runs
    assert len(n_lines) == 1 + 4
    # Run again, should not duplicate
    rep.run_all()
    n_lines2 = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(n_lines2) == len(n_lines)

