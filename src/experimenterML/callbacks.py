from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class Callback:
    """Base callback with no-ops."""

    def on_run_begin(self, params: Dict[str, Any], run_id: str) -> None:  # pragma: no cover
        pass

    def on_run_end(self, params: Dict[str, Any], run_id: str, metrics: Dict[str, Any]) -> None:  # pragma: no cover
        pass


@dataclass
class EarlyStopping(Callback):
    patience: int = 5
    best: float = float("inf")
    bad: int = 0

    def update(self, current: float) -> bool:
        if current < self.best:
            self.best = current
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience

