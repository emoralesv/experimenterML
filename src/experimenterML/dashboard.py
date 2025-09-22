from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

try:
    # Only import when used via CLI entry, avoid circulars when imported from Experiment
    from . import Experiment  # type: ignore
except Exception:  # pragma: no cover
    Experiment = object  # fallback type for type checkers


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="tests/exp_example.yaml")
    parser.add_argument("--refresh", type=float, default=0.0, help="Auto-refresh interval in seconds")
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _resolve_out_csv(config_path: Path | None, out_csv: str) -> Path:
    out = Path(out_csv)
    if out.is_absolute() or config_path is None:
        return out.resolve()
    return (config_path.parent / out).resolve()


def render_dashboard(exp, config_path: Path | None, live_refresh: bool, refresh_sec: int) -> None:
    """Render the Streamlit dashboard for a given Experiment instance.

    This function can be called programmatically from Experiment.
    """
    st.set_page_config(page_title="experimenterML Dashboard", layout="wide")

    reps = exp.reps
    reps.validate()

    # Compute grid dataframe: uid + params
    param_keys: List[str] = list(exp.config.params.keys())
    grid_rows = [
        {"uid": r.uid, **{k: r.params.get(k) for k in param_keys}}
        for r in reps.experiment_dicts
    ]
    df_grid = pd.DataFrame(grid_rows) if grid_rows else pd.DataFrame(columns=["uid", *param_keys])

    out_csv_path = exp.config.out_csv
    df_results_full = _read_csv(out_csv_path)

    # Determine metrics columns (exclude uid and param keys)
    metric_cols = [c for c in df_results_full.columns if c not in {"uid", *param_keys}]
    cols_to_merge = ["uid", *metric_cols] if metric_cols else ["uid"]
    df = df_grid.merge(df_results_full[cols_to_merge], on="uid", how="left")

    # Status and progress
    done_set = set(reps.done_exps)
    df["status"] = df["uid"].apply(lambda u: "done" if u in done_set else "pending")

    total = len(df)
    done = int((df["status"] == "done").sum())
    pending = total - done

    st.header(f"Experiments: {exp.config.module}.{exp.config.function}")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
    with c1:
        st.progress(0 if total == 0 else done / total)
    with c2:
        st.metric("Total", total)
    with c3:
        st.metric("Done", done)
    with c4:
        st.metric("Pending", pending)

    # Sidebar filters
    st.sidebar.title("experimenterML Dashboard")
    st.sidebar.subheader("Filters")
    mask = pd.Series([True] * len(df))
    for k in param_keys:
        vals = sorted(df[k].dropna().unique().tolist())
        if not vals:
            continue
        selected = st.sidebar.multiselect(k, vals, default=vals)
        mask &= df[k].isin(selected)

    status_sel = st.sidebar.multiselect("status", ["done", "pending"], default=["done", "pending"])
    mask &= df["status"].isin(status_sel)

    df_view = df[mask].copy()
    # Order columns: status, uid, params..., metrics...
    ordered_cols = ["status", "uid", *param_keys, *metric_cols]
    existing_cols = [c for c in ordered_cols if c in df_view.columns]
    df_view = df_view[existing_cols]

    st.subheader("Results table")
    st.dataframe(df_view, width="content", height=420)

    # Quick chart: pick X param and Y metric
    st.subheader("Quick chart")
    if metric_cols:
        x_param = st.selectbox("Group by (X)", options=param_keys, index=0 if param_keys else None)
        y_metric = st.selectbox("Metric (Y)", options=metric_cols, index=0)
        if x_param and y_metric:
            agg = df_view.dropna(subset=[y_metric]).groupby(x_param)[y_metric].mean().reset_index()
            st.bar_chart(agg, x=x_param, y=y_metric, width="container")
    else:
        st.info("No numeric metrics yet. Once results append metrics to the CSV, a chart will appear here.")

    # Pending preview
    st.subheader("Pending experiments (sample)")
    pending_df = df[df["status"] == "pending"][ ["uid", *param_keys] ]
    st.dataframe(pending_df.head(20), width=True, height=200)

    st.caption(f"CSV: {out_csv_path}")

    # Auto-refresh loop (single-hop): sleep then rerun
    if live_refresh:
        time.sleep(max(1, int(refresh_sec)))

        st.rerun()


def run_with_experiment(exp, refresh: float = 0.0) -> None:
    """Public entry to run dashboard with an existing Experiment instance."""
    config_path = getattr(exp, "config_path", None)
    live = refresh and refresh > 0
    render_dashboard(exp, config_path, live_refresh=True, refresh_sec=5)


def main(config:str):
    # CLI entry when executed as a module/script
    args = _parse_args()
    from experimenterML import Experiment as x
    config =  getattr(args, "config", config)
    exp = x.Experimenter(config)
    # UI controls in the sidebar control live refresh when run via CLI
    st.sidebar.caption("Use the Run CLI separately. This dashboard reads the CSV and updates here.")
    render_dashboard(exp, getattr(exp, "config_path", None), live_refresh=True, refresh_sec=5)


if __name__ == "__main__":
    main(config="tests/exp_example.yaml")
