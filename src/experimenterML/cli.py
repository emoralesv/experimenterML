from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_from_config


def _run(args: argparse.Namespace) -> None:
    run_from_config(args.config)


def _ls(args: argparse.Namespace) -> None:
    path = Path(args.results)
    if not path.exists():
        print("No results.")
        return
    print(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp", description="experimenterML CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run experiments from YAML config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_run)

    p_ls = sub.add_parser("ls", help="Show results CSV")
    p_ls.add_argument("--results", default="results/experiments.csv")
    p_ls.set_defaults(func=_ls)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

