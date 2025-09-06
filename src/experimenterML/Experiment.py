import os
import sys
from dataclasses import dataclass
from itertools import product
import streamlit.web.cli as stcli
import numpy as np
import hashlib
import json
import pandas as pd
from typing import Any, Callable, Dict
import yaml
import importlib
from pathlib import Path
import csv
import sys
import subprocess
import streamlit.web.bootstrap
class Experimenter:
    def __init__(self, config:str):
        self.config_path = Path(config)
        self.config = _ExpConfig.from_yaml(config)
        self.reps = _Repetitions(self.config.params,self.config.out_csv)

    def run(self, run_dashboard: bool = True):
        if run_dashboard:
            self.dashboard()
        print(f"Running experiment: {self.config.function}")
        while self.reps.next() is not None:
            current_exp = self.reps.currentExperiment
            self.reps.print(current=True)
            result = self.config.evaluation_fn(**current_exp.params)
            if result is not None:
                with open(self.config.out_csv, "a", encoding="utf-8",newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["uid"] + list(result.keys()))
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow({"uid": current_exp.uid, **result}) 
                
        


    def printExperiments(self, current=False):
        print(f"Experiments for function: {self.config.function}")
        self.reps.print(current=current)

    def dashboard(self, refresh: float = 0.0):
        from . import dashboard as _dash
        here = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(here, "dashboard.py")
        cmd = [sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true" if True else "false",]
        process = subprocess.Popen(cmd)

        


@dataclass
class _ExpConfig:
    name: str
    module: str
    function: str
    out_csv: str
    params: Dict[str, Any]
    evaluation_fn: Callable = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        name = data.get("name", "default_experiment")
        module = data.get("module")
        function = data.get("function")
        out_csv = data.get("out_csv", "results.csv")
        params = data.get("params", {})
        if not module or not function:
            raise ValueError("YAML must contain 'module' and 'function' fields.")
        # Ensure imports work regardless of current working directory.
        # Add the config dir and its parent (project root) to sys.path if missing.
        try:
            cfg_path = Path(path).resolve()
            cfg_dir = cfg_path.parent
            candidates = [cfg_dir, cfg_dir.parent]
            for p in candidates:
                sp = str(p)
                if p.is_dir() and sp not in sys.path:
                    sys.path.insert(0, sp)
        except Exception:
            pass
        try:
            mod = importlib.import_module(module)
            evaluation_fn = getattr(mod, function)
        except Exception as e:
            raise ImportError(f"Could not import {function} from {module}: {e}")
        return cls(name, module, function, out_csv, params, evaluation_fn)

    def print(self):
        print(f"Experiment config:")
        print(f"  module: {self.module}")
        print(f"  function: {self.function}")
        print(f"  out_csv: {self.out_csv}")
        print(f"  params: {self.params}")
        



@dataclass
class _Repetitions:
    @dataclass
    class _rep:
        params: Dict[str, Any]
        uid: str = None
        
    def __init__(self, params,results_csv):
        self.results_csv = results_csv
        self.params = params
        self.experiment_dicts = []
        # Build all combinations ONCE
        keys = list(params.keys())
        values = [v if isinstance(v, list) else [v] for v in params.values()]
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            uid = self.hash_experiment(param_dict)
            new_exp = self._rep(param_dict,uid)
            self.experiment_dicts.append(new_exp)

        # After you finish the outer loop over all `conf` items (i.e., after building experiment_dicts):
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
                for k, v in d.params.items():
                    print(f"   {k}: {v}")
                print("-" * 40)

    def next(self):
        self.validate()
        for exp in self.experiment_dicts:
            if exp.uid not in self.done_exps:
                self.currentExperiment = exp
                return exp
        return None


    def hash_experiment(self, exp_dict):
        exp_str = json.dumps(exp_dict, sort_keys=True)
        return hashlib.sha256(exp_str.encode()).hexdigest()[:16]  
    def validate(self):
        """Load completed uids from results.csv if present."""
        if os.path.exists(self.results_csv):
            df_results = pd.read_csv(self.results_csv)
            if "uid" in df_results.columns:
                self.done_exps = set(df_results["uid"].astype(str).tolist())
            else:
                self.done_exps = set()
        else:
            self.done_exps = set()
