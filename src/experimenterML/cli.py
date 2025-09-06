from __future__ import annotations

import argparse
from pathlib import Path

from .Experiment import Experimenter


def _run(args: argparse.Namespace) -> None:
    exp = Experimenter(args.config)
    exp.run()


def _ls(args: argparse.Namespace) -> None:
    path = Path(args.results)
    if not path.exists():
        print("No results.")
        return
    print(path.read_text(encoding="utf-8"))


def _ui(args: argparse.Namespace) -> None:
    """Render a simple table of experiments and their status.

    Status is computed by comparing each experiment's uid against the
    uids found in the CSV defined in the YAML (out_csv).
    """
    exp = Experiment(args.config)
    reps = exp.reps

    # Ensure we have up-to-date completion info
    reps.validate()

    # Build headers dynamically from parameter keys
    param_keys = list(exp.config.params.keys())
    headers = ["uid", "status", *param_keys]

    # Compute rows
    rows = []
    for r in reps.experiment_dicts:
        status = "done" if r.uid in reps.done_exps else "pending"
        row = [r.uid, status] + [str(r.params.get(k, "")) for k in param_keys]
        rows.append(row)

    # Pretty-print as a minimal fixed-width table
    col_widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]

    def fmt_row(values):
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in col_widths)

    print(f"Experiments for: {exp.config.module}.{exp.config.function}")
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))

    # Summary
    total = len(rows)
    done = sum(1 for r in rows if r[1] == "done")
    print()
    print(f"Total: {total} | Done: {done} | Pending: {total - done}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp", description="experimenterML CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run experiments from YAML config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_run)

    p_ls = sub.add_parser("ls", help="Show results CSV")
    p_ls.add_argument("--results", default="results.csv")
    p_ls.set_defaults(func=_ls)

    # UI: show all experiments defined by the YAML as a table
    p_ui = sub.add_parser("ui", help="Show experiment grid and status")
    p_ui.add_argument("--config", required=True, help="Path to YAML config")
    p_ui.set_defaults(func=_ui)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
