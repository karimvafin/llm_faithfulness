from collections import defaultdict
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _legend_marker_name(raw: str) -> str:
    name = raw.strip().strip("=")
    return name or raw.strip()


def _section_color(section: str) -> str:
    palette = {
        "SKELETON": "#4c78a8",
        "SCHEMA_LINKS": "#f58518",
        "SLOT_MATCHING": "#54a24b",
        "SQL": "#e45756",
    }
    return palette.get(section, "#777777")


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
    # ax.set_ylim(0, 1)
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
    # ax.set_ylim(0, 1)
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


def plot_fstrong_vs_intervention_level(
    reports: Iterable[dict],
    out_path: str,
    *,
    mode: str | None = "structure_prediction",
) -> None:
    """Line plot for thesis figure: x = intervention level, y = FStrong, one line per model."""
    by_model: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in reports:
        if mode is not None and r.get("mode") != mode:
            continue
        model = str(r.get("model", "?"))
        level = int(r.get("intervention_level", 0))
        by_model[model][level].append(float(r.get("faithfulness_strong", 0.0)))

    if not by_model:
        raise ValueError("No reports left after filtering; check --mode/inputs.")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, points in sorted(by_model.items()):
        levels = sorted(points.keys())
        vals = [float(np.mean(points[level])) for level in levels]
        short_model = model.split("/")[-1]
        ax.plot(levels, vals, marker="o", linewidth=2, label=short_model)

    ax.set_xlabel("Intervention level")
    ax.set_ylabel(r"$F_{Strong}$")
    # ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_performance_fid_bars(
    reports: Iterable[dict],
    out_path: str,
    *,
    mode: str | None = "structure_prediction",
    intervention_level: int = 0,
) -> None:
    """Grouped bars per model with Performance and F_ID side-by-side."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        if mode is not None and r.get("mode") != mode:
            continue
        if int(r.get("intervention_level", 0)) != intervention_level:
            continue
        by_model[str(r.get("model", "?"))].append(r)

    if not by_model:
        raise ValueError("No reports left after filtering; check --mode/--intervention-level/inputs.")

    models = sorted(by_model.keys())
    perf = [float(np.mean([x.get("performance", 0.0) for x in by_model[m]])) for m in models]
    fid = [float(np.mean([x.get("faithfulness_id", 0.0) for x in by_model[m]])) for m in models]
    labels = [m.split("/")[-1] for m in models]

    x = np.arange(len(models), dtype=float)
    width = 0.38
    fig_w = max(7.2, 1.2 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))
    ax.bar(x - width / 2, perf, width=width, label="Performance", color="#4c78a8")
    ax.bar(x + width / 2, fid, width=width, label=r"$F_{ID}$", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
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
    drawn_region_labels: set[str] = set()
    min_tokens = min((len(item[1]) for item in series if item[1]), default=0)
    for item in series:
        label = item[0]
        entropies = item[1]
        style: dict = item[2] if len(item) >= 3 else {}
        markers: dict[str, int] | None = item[3] if len(item) >= 4 else None

        if min_tokens <= 0:
            continue
        ys = np.asarray(entropies[:min_tokens], dtype=float)
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
            sorted_markers = sorted(
                ((_legend_marker_name(name), int(pos)) for name, pos in markers.items()),
                key=lambda kv: kv[1],
            )
            # Fill each mediator section as a vertical band [start, next_start),
            # then draw colored boundaries so section splits are easy to read.
            for idx, (section_name, start_pos) in enumerate(sorted_markers):
                if start_pos >= min_tokens:
                    continue
                section_color = _section_color(section_name)
                end_pos = sorted_markers[idx + 1][1] if idx + 1 < len(sorted_markers) else min_tokens
                end_pos = min(end_pos, min_tokens)
                if end_pos > start_pos:
                    region_label = section_name if section_name not in drawn_region_labels else "_nolegend_"
                    ax.axvspan(
                        start_pos,
                        end_pos,
                        color=section_color,
                        alpha=0.12,
                        label=region_label,
                        zorder=0,
                    )
                    if region_label != "_nolegend_":
                        drawn_region_labels.add(section_name)
                ax.axvline(
                    start_pos,
                    color=section_color,
                    linestyle=":",
                    linewidth=1.2,
                    zorder=1,
                )

    ax.set_xlabel("Token index")
    ax.set_ylabel("Entropy (nats)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup: dict[str, object] = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        ax.legend(
            dedup.values(),
            dedup.keys(),
            fontsize=9,
            loc="best",
            frameon=True,
            framealpha=0.92,
            fancybox=True,
            borderpad=0.5,
            handlelength=2.8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
