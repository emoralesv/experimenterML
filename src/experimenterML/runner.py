from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

from .repetitions import Repetitions


@dataclass
class ExpConfig:
    module: str
    function: str
    out_dir: str
    params: Dict[str, Any]


def load_config(path: str | Path) -> ExpConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExpConfig(
        module=data["module"], function=data["function"], out_dir=data.get("out_dir", "results"), params=data["params"]
    )


def load_train_fn(module: str, function: str) -> Callable[[Dict, str], Dict]:
    mod = importlib.import_module(module)
    fn = getattr(mod, function)
    return fn


def run_from_config(path: str | Path) -> None:
    cfg = load_config(path)
    train_fn = load_train_fn(cfg.module, cfg.function)
    keys = sorted(cfg.params.keys())
    rep = Repetitions(train_fn=train_fn, param_grid=cfg.params, hash_keys=keys, out_dir=cfg.out_dir)
    rep.run_all()

