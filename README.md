# experimenterML

Lightweight framework to define, run, and reproduce experiments.

- Grid/combinatorial sweeps
- Hashing (sha256 → 8-char uid) for configs
- Results logging to CSV/JSON; resume/skip existing runs
- CLI: `exp run --config configs/exp_example.yaml`, `exp ls`

## Install

```
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```python
from experimenterML.repetitions import Repetitions

def train_fn(params: dict, run_id: str) -> dict:
    # Your training here; return metrics
    return {"accuracy": 0.9, "f1": 0.88, "loss": 0.3}

rep = Repetitions(
    train_fn=train_fn,
    param_grid={"lr": [1e-3, 1e-4], "bs": [32, 64], "seed": [0, 1]},
    hash_keys=["lr", "bs", "seed"],
    out_dir="results",
)
rep.run_all()
```

## CLI

```
exp run --config configs/exp_example.yaml
exp ls --results results/experiments.csv
```

## Config example

```yaml
module: mypkg.myexp
function: train
out_dir: results
params:
  lr: [0.001, 0.0001]
  bs: [32, 64]
  seed: [0, 1]
```

