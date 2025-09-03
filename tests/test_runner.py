from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from experimenterML.runner import run_from_config


def test_runner_with_temp_module(tmp_path: Path) -> None:
    # Create a temporary module with a train function
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("\n")
    (pkg_dir / "myexp.py").write_text(
        dedent(
            """
            def train(params, run_id):
                return {"accuracy": 0.5, "f1": 0.5, "loss": 1.0}
            """
        )
    )
    sys.path.insert(0, tmp_path.as_posix())
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        dedent(
            f"""
            module: mypkg.myexp
            function: train
            out_dir: { (tmp_path / 'res').as_posix() }
            params:
              a: [1, 2]
            """
        )
    )
    run_from_config(cfg)
    csv_path = tmp_path / "res" / "experiments.csv"
    assert csv_path.exists()

