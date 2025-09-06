import os
from dataclasses import dataclass
from itertools import product
import numpy as np
import hashlib
import json
import pandas as pd
from typing import Any, Callable, Dict
import yaml
import importlib
from pathlib import Path

@dataclass
class ExpConfig:
    module: str
    function: str
    out_dir: str
    params: Dict[str, Any]
    evaluation_fn: Callable = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        module = data.get("module")
        function = data.get("function")
        out_dir = data.get("out_dir", "results")
        params = data.get("params", {})
        if not module or not function:
            raise ValueError("YAML must contain 'module' and 'function' fields.")
        try:
            mod = importlib.import_module(module)
            evaluation_fn = getattr(mod, function)
        except Exception as e:
            raise ImportError(f"Could not import {function} from {module}: {e}")
        return cls(module, function, out_dir, params, evaluation_fn)

    def print(self):
        print(f"Experiment config:")
        print(f"  module: {self.module}")
        print(f"  function: {self.function}")
        print(f"  out_dir: {self.out_dir}")
        print(f"  params: {self.params}")
        



@dataclass
class Repetitions:
    def __init__(self, config,path="results", name="results.csv"):


        self.path = path
        self.name = name
        os.makedirs(self.path, exist_ok=True)
        self.results_path = os.path.join(self.path, self.name)
        self.experiment_dicts = []
        for conf in config:
            
            keys = list(conf.keys())
            values = [v if isinstance(v, list) else [v] for v in conf.values()]
            for combo in product(*values):
                param_dict = dict(zip(keys, combo))
                param_dict["uid"] = self.hash_experiment(param_dict)
                self.experiment_dicts.append(param_dict)

        self.realized = np.full(len(self.experiment_dicts), False)
        self.current = None
        self.currentExperiment = None
        
    def __len__(self):
        return len(self.experiment_dicts)

    def print(self, current=False):
        if current:
            experiments = [self.currentExperiment]
        else:
            experiments = self.experiment_dicts
        if experiments:
            for d in experiments:
                for k, v in d.items():
                    print(f"   {k}: {v}")
                print("-" * 40)

    def next(self):
        for i, exp in enumerate(self.experiment_dicts):
            if not self.realized[i]:
                self.current = i
                self.currentExperiment = exp
                return exp
        return None

    def realize(self):
        if self.current is not None:
            self.realized[self.current] = True

    def hash_experiment(self, exp_dict):
        exp_str = json.dumps(exp_dict, sort_keys=True)
        return hashlib.sha256(exp_str.encode()).hexdigest()[:8]  
    def validate(self):
        if os.path.exists(self.results_path):
            df_results = pd.read_csv(self.results_path)
            done_exps = set(df_results["uid"])
        else:
            df_results = pd.DataFrame()
            done_exps = set()
