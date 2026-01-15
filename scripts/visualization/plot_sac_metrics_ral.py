#!/usr/bin/env python3
"""Plot SAC training curves from metrics.csv for RA-L/IEEE-style figures.

Reads one or more run directories produced by scripts/training/train_sac_diffusion_simple.py.
Outputs vector PDF (and optional PNG) figures suitable for LaTeX inclusion.

Example:
  python scripts/visualization/plot_sac_metrics_ral.py \
    --run_dirs experiments/sac_training/sac_training_20260112_070517

Or plot multiple runs:
  python scripts/visualization/plot_sac_metrics_ral.py \
    --runs_glob 'experiments/sac_training/sac_training_*/metrics.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclass
class Series:
    step: np.ndarray
    values: Dict[str, np.ndarray]


@dataclass
class GroupedSeries:
    label: str
    n_runs: int
    step: np.ndarray
    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]


def _read_metrics_csv(csv_path: Path) -> Series:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def col(name: str, dtype=float) -> np.ndarray:
        out: List[float] = []
        for r in rows:
            v = r.get(name, "")
            if v is None or v == "":
                out.append(np.nan)
            else:
                out.append(dtype(v))
        return np.asarray(out)

    step = col("step", int).astype(np.int64)

    values = {
        "success_rate": col("success_rate", float),
        "avg_reward_10": col("avg_reward_10", float),
        "avg_length_10": col("avg_length_10", float),
        "critic1_loss": col("critic1_loss", float),
        "critic2_loss": col("critic2_loss", float),
        "actor_loss": col("actor_loss", float),
        "q_value": col("q_value", float),
        "alpha": col("alpha", float),
        "episodes": col("episodes", int).astype(np.int64),
        "buffer_size": col("buffer_size", int).astype(np.int64),
        "wall_time_sec": col("wall_time_sec", float),
    }
    return Series(step=step, values=values)


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return x
    if alpha >= 1.0:
        return x

    y = np.empty_like(x, dtype=float)
    y[:] = np.nan

    prev = None
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
            continue
        if prev is None:
            prev = float(v)
        else:
            prev = alpha * float(prev) + (1.0 - alpha) * float(v)
        y[i] = prev
    return y


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _interp_to_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Interpolate y(x) onto x_grid. Out-of-range -> NaN."""
    x_f, y_f = _finite_xy(x, y)
    if x_f.size < 2:
        out = np.full_like(x_grid, np.nan, dtype=float)
        if x_f.size == 1:
            out[x_grid == x_f[0]] = y_f[0]
        return out

    order = np.argsort(x_f)
    x_f = x_f[order]
    y_f = y_f[order]

    y_i = np.interp(x_grid, x_f, y_f)
    y_i[(x_grid < x_f[0]) | (x_grid > x_f[-1])] = np.nan
    return y_i


def _group_label_from_dir(run_dir_name: str, group_regex: str) -> str:
    m = re.match(group_regex, run_dir_name)
    if m is None:
        return run_dir_name
    if "group" in m.groupdict():
        return str(m.group("group"))
    # fallback to first capturing group
    if m.groups():
        return str(m.groups()[0])
    return run_dir_name


def _aggregate_groups(
    metrics_paths: List[Path],
    series_list: List[Series],
    group_regex: str,
    x_grid_step: int,
) -> List[GroupedSeries]:
    by_label: Dict[str, List[Series]] = {}
    for p, s in zip(metrics_paths, series_list, strict=False):
        label = _group_label_from_dir(p.parent.name, group_regex)
        by_label.setdefault(label, []).append(s)

    grouped: List[GroupedSeries] = []
    for label, runs in sorted(by_label.items(), key=lambda kv: kv[0]):
        max_step = int(np.nanmax(np.concatenate([r.step for r in runs])))
        x_grid = np.arange(0, max_step + 1, int(x_grid_step), dtype=np.int64)
        if x_grid.size == 0:
            x_grid = np.asarray([0], dtype=np.int64)

        keys = list(runs[0].values.keys())
        mean: Dict[str, np.ndarray] = {}
        std: Dict[str, np.ndarray] = {}

        for k in keys:
            ys = np.stack(
                [
                    _interp_to_grid(
                        r.step.astype(float),
                        r.values[k].astype(float),
                        x_grid.astype(float),
                    )
                    for r in runs
                ],
                axis=0,
            )

            # Robust aggregation: avoid warnings when all-NaN at a step.
            count = np.sum(np.isfinite(ys), axis=0)
            m = np.full((ys.shape[1],), np.nan, dtype=float)
            s = np.full((ys.shape[1],), np.nan, dtype=float)
            has_any = count > 0
            if np.any(has_any):
                m[has_any] = np.nanmean(ys[:, has_any], axis=0)
            has_two = count > 1
            if np.any(has_two):
                s[has_two] = np.nanstd(ys[:, has_two], axis=0)

            mean[k] = m
            std[k] = s

        grouped.append(
            GroupedSeries(label=label, n_runs=len(runs), step=x_grid, mean=mean, std=std)
        )

    return grouped


def _configure_matplotlib(fontsize: int = 8) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "legend.fontsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _single_column_fig(figsize: Tuple[float, float] = (3.5, 2.3)) -> tuple[plt.Figure, plt.Axes]:
    # IEEE single-column width ~3.5in
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _set_color_cycler(n: int) -> List[str]:
    # Matplotlib tab10 is reasonably printer-friendly.
    cmap = plt.get_cmap("tab10")
    colors = [mpl.colors.to_hex(cmap(i % 10)) for i in range(max(n, 1))]
    return colors


def _plot_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
    alpha_fill: float = 0.18,
) -> None:
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    # Only shade where std is finite.
    if std is not None:
        lo = mean - std
        hi = mean + std
        mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(x)
        if np.any(mask):
            ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=alpha_fill, linewidth=0)


def _save(fig: plt.Figure, out_base: Path, write_png: bool) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    if write_png:
        fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot SAC training curves from metrics.csv")
    parser.add_argument(
        "--run_dirs",
        nargs="*",
        default=None,
        help="One or more run directories (each must contain metrics.csv)",
    )
    parser.add_argument(
        "--runs_glob",
        type=str,
        default="experiments/sac_training/sac_training_*/metrics.csv",
        help="Glob to discover metrics.csv files when --run_dirs not provided",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: run_dir/figures for single run, else figures/ral)",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.90,
        help="EMA smoothing factor in [0,1). 0 disables.",
    )
    parser.add_argument(
        "--group_regex",
        type=str,
        default=r"(?P<group>.*)_\d{8}_\d{6}$",
        help="Regex to group multiple seed runs into one method label. Default groups by stripping _YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--x_grid_step",
        type=int,
        default=1000,
        help="Resample step size for mean±std aggregation.",
    )
    parser.add_argument(
        "--fig_w",
        type=float,
        default=3.5,
        help="Figure width in inches (default: IEEE single-column ~3.5in)",
    )
    parser.add_argument(
        "--fig_h",
        type=float,
        default=2.3,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--write_png",
        action="store_true",
        help="Also save PNG copies (PDF is always saved)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix",
    )

    args = parser.parse_args(argv)

    # Resolve runs
    metrics_paths: List[Path] = []
    if args.run_dirs:
        for rd in args.run_dirs:
            p = Path(rd) / "metrics.csv"
            metrics_paths.append(p)
    else:
        metrics_paths = [Path(p) for p in glob.glob(args.runs_glob)]

    metrics_paths = [p for p in metrics_paths if p.exists()]
    if not metrics_paths:
        raise SystemExit("No metrics.csv found. Provide --run_dirs or adjust --runs_glob.")

    # Print what we are going to plot (avoid confusion about glob matching / image caching)
    print(f"[plot] matched metrics.csv: {len(metrics_paths)}")
    for p in metrics_paths:
        try:
            n_rows = sum(1 for _ in open(p, "r", encoding="utf-8"))
        except Exception:
            n_rows = -1
        print(f"[plot] - {p} (rows={n_rows})")
    if len(metrics_paths) == 1:
        print("[plot] note: only 1 run matched, so mean±std shading will look like a single curve.")

    run_labels = [p.parent.name for p in metrics_paths]
    series_list = [_read_metrics_csv(p) for p in metrics_paths]

    # Aggregate multi-seed runs (mean±std)
    grouped = _aggregate_groups(
        metrics_paths=metrics_paths,
        series_list=series_list,
        group_regex=args.group_regex,
        x_grid_step=args.x_grid_step,
    )

    print(f"[plot] groups: {len(grouped)} (group_regex={args.group_regex!r})")
    for g in grouped:
        # show a quick sanity signal: how many non-NaN points exist for return
        n_pts = int(np.sum(np.isfinite(g.mean.get("avg_reward_10", np.asarray([])))))
        print(f"[plot] - {g.label}: n_runs={g.n_runs}, return_points={n_pts}")
    colors = _set_color_cycler(len(grouped))

    # Decide output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (metrics_paths[0].parent / "figures") if len(metrics_paths) == 1 else Path("figures/ral")

    _configure_matplotlib()

    fig_size = (float(args.fig_w), float(args.fig_h))

    # 1) Average reward (last-10)
    fig, ax = _single_column_fig(figsize=fig_size)
    for color, g in zip(colors, grouped, strict=False):
        y_mean = _ema(g.mean["avg_reward_10"], args.ema)
        y_std = g.std["avg_reward_10"]
        label = f"{g.label} (n={g.n_runs})"
        _plot_mean_std(ax, g.step, y_mean, y_std, label=label, color=color)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Avg episode return (last 10)")
    if args.title:
        ax.set_title(f"{args.title} Return")
    ax.legend(loc="best", frameon=False)
    _save(fig, out_dir / "sac_return", args.write_png)
    plt.close(fig)

    # 2) Success rate
    fig, ax = _single_column_fig(figsize=fig_size)
    for color, g in zip(colors, grouped, strict=False):
        y_mean = _ema(g.mean["success_rate"], args.ema)
        y_std = g.std["success_rate"]
        label = f"{g.label} (n={g.n_runs})"
        _plot_mean_std(ax, g.step, y_mean, y_std, label=label, color=color)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    if args.title:
        ax.set_title(f"{args.title} Success")
    ax.legend(loc="best", frameon=False)
    _save(fig, out_dir / "sac_success", args.write_png)
    plt.close(fig)

    # 3) Loss curves (RA-L style: color=method, linestyle=metric)
    fig, ax = _single_column_fig(figsize=fig_size)
    for color, g in zip(colors, grouped, strict=False):
        critic_mean = 0.5 * (g.mean["critic1_loss"] + g.mean["critic2_loss"])
        critic_std = 0.5 * (g.std["critic1_loss"] + g.std["critic2_loss"])
        actor_mean = g.mean["actor_loss"]
        actor_std = g.std["actor_loss"]
        base = f"{g.label} (n={g.n_runs})"

        _plot_mean_std(
            ax,
            g.step,
            _ema(critic_mean, args.ema),
            critic_std,
            label=f"{base} critic",
            color=color,
            linestyle="-",
        )
        _plot_mean_std(
            ax,
            g.step,
            _ema(actor_mean, args.ema),
            actor_std,
            label=f"{base} actor",
            color=color,
            linestyle="--",
        )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    if args.title:
        ax.set_title(f"{args.title} Loss")
    ax.legend(loc="best", frameon=False, ncol=1)
    _save(fig, out_dir / "sac_losses", args.write_png)
    plt.close(fig)

    # 4) Q value
    fig, ax = _single_column_fig(figsize=fig_size)
    for color, g in zip(colors, grouped, strict=False):
        y_mean = _ema(g.mean["q_value"], args.ema)
        y_std = g.std["q_value"]
        label = f"{g.label} (n={g.n_runs})"
        _plot_mean_std(ax, g.step, y_mean, y_std, label=label, color=color)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Q estimate")
    if args.title:
        ax.set_title(f"{args.title} Q")
    ax.legend(loc="best", frameon=False)
    _save(fig, out_dir / "sac_q", args.write_png)
    plt.close(fig)

    print(f"✓ Saved figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
