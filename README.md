# experimenterML

Lightweight experiment runner for combinatorial sweeps with resume-by-UID and CSV logging.

- Grid/combinatorial parameter sweeps
- Stable 8-char UID per config (sha256 of params)
- Append-only CSV logging with automatic resume/skip of completed runs
- Simple CLI with a text UI to list experiment grid and status

## Install

Option A — editable install (recommended during development):

```
pip install -e .
```

Option B — run without installing (from project root):

```
python -m experimenterML.cli --help
```

## Define your evaluation function

Create a Python module with a function that accepts your parameters and returns a dict of metrics.

Example: `tests/example_module.py`

```python
import random

def random_evaluation(**kwargs):
    # Do any computation here; return a dict of results
    accuracy = round(random.uniform(0.5, 1.0), 3)
    return {**kwargs, "accuracy": accuracy}
```

## YAML config

Point to the module/function and list parameter values (lists produce a full grid).

Example: `tests/exp_example.yaml`

```yaml
name: example_experiment
module: tests.example_module   # Python import path
function: random_evaluation    # Function name in the module
out_csv: tests/results.csv     # Where to append results
params:
  models: ["resnet50", "vgg16"]
  size: [128, 256]
```

You can add more parameters (lists or scalars); the grid combines all lists.

## Run from the CLI

If installed (via `pip install -e .`):

```
exp run --config tests/exp_example.yaml     # run the sweep
exp ui  --config tests/exp_example.yaml     # show grid + status
exp ls  --results tests/results.csv         # print the CSV
```

Without installing (from the repo root):

```
python -m experimenterML.cli run --config tests/exp_example.yaml
python -m experimenterML.cli ui  --config tests/exp_example.yaml
python -m experimenterML.cli ls  --results tests/results.csv
```

## Programmatic usage

```python
from experimenterML.Experiment import Experiment

exp = Experiment("tests/exp_example.yaml")
exp.run()                      # executes pending runs only
exp.printExperiments()         # prints the experiment parameters
```

## How it resumes

- Each parameter combination is hashed to a stable 8-char `uid`.
- On each run, the tool reads `out_csv` (if it exists) and skips any `uid` already present.
- Results are appended to the CSV with columns `uid` plus whatever your function returns.

## Notes & tips

- Ensure `module` in YAML is a valid import path from where you run the CLI (e.g., `tests.example_module`).
- Run commands from the project root so your modules are importable.
- The UI table reflects status based on `out_csv`: if it does not exist, all are pending.

