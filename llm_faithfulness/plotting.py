from collections import defaultdict
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_faithfulness_vs_level(reports: Iterable[dict], out_path: str) -> None:
    """Line plot: x = intervention_level, y = faithfulness_strong. One line per (model, mode)."""
    by_series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in reports:
        by_series[(r["model"], r["mode"])].append((r["intervention_level"], r["faithfulness_strong"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    for (model, mode), points in sorted(by_series.items()):
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=f"{model} ({mode})")
    ax.set_xlabel("Intervention level")
    ax.set_ylabel("faithfulness_strong")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_bars(reports: Iterable[dict], metric: str, out_path: str) -> None:
    """Grouped bars: one bar per (model, mode) at the report's intervention_level."""
    items = sorted(reports, key=lambda r: (r["model"], r["mode"], r["intervention_level"]))
    labels = [f'{r["model"]}\n{r["mode"]}\nL{r["intervention_level"]}' for r in items]
    values = [r[metric] for r in items]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(items)), 4))
    ax.bar(range(len(items)), values)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_perf_vs_faithfulness(reports: Iterable[dict], out_path: str) -> None:
    """Scatter: x = faithfulness_strong, y = performance. One point per report."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in reports:
        ax.scatter(r["faithfulness_strong"], r["performance"])
        ax.annotate(
            f'{r["model"]} L{r["intervention_level"]}',
            (r["faithfulness_strong"], r["performance"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("faithfulness_strong")
    ax.set_ylabel("performance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_token_entropy_vs_index(
    series: Sequence[tuple],
    out_path: str,
    *,
    smoothing_window: int = 1,
    title: str | None = None,
) -> None:
    """Line plot: x = generated-token index, y = Shannon entropy (nats).
    Each item in `series` is one of:
      (label, entropies)
      (label, entropies, style)              # style: dict of matplotlib plot kwargs
      (label, entropies, style, markers)     # markers: dict[str, int] of vline labels -> token index
    Marker vlines are drawn in the same color as their series with linestyle ':'."""
    fig, ax = plt.subplots(figsize=(8, 4))
    drawn_marker_labels: set[str] = set()
    for item in series:
        label = item[0]
        entropies = item[1]
        style: dict = item[2] if len(item) >= 3 else {}
        markers: dict[str, int] | None = item[3] if len(item) >= 4 else None

        ys = np.asarray(entropies, dtype=float)
        if smoothing_window > 1 and ys.shape[0] >= smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            ys_plot = np.convolve(ys, kernel, mode="valid")
            offset = (smoothing_window - 1) // 2
            xs = np.arange(offset, offset + ys_plot.shape[0])
        else:
            ys_plot = ys
            xs = np.arange(ys.shape[0])
        line, = ax.plot(xs, ys_plot, label=label, **style)

        if markers:
            color = line.get_color()
            for marker_name, pos in markers.items():
                ax.axvline(pos, color=color, linestyle=":", alpha=0.5, linewidth=1)
                ann_label = marker_name if marker_name not in drawn_marker_labels else None
                ax.annotate(
                    marker_name,
                    xy=(pos, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(2, -10),
                    textcoords="offset points",
                    fontsize=6,
                    rotation=90,
                    color=color,
                    alpha=0.7,
                )
                if ann_label:
                    drawn_marker_labels.add(marker_name)

    ax.set_xlabel("Token index")
    ax.set_ylabel("Entropy (nats)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
