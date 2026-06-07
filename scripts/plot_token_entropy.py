import argparse
import json
import os
import statistics

from llm_faithfulness.plotting import plot_token_entropy_vs_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-token Shannon entropy from a saved evaluation report. "
        "The report must have been produced by scripts/evaluate.py with --save-entropies."
    )
    p.add_argument("--report", required=True, help="Path to evaluation report JSON")
    p.add_argument(
        "--field",
        choices=["completion", "intervened_completion", "both"],
        default="both",
        help="Which generation's entropies to plot. 'both' overlays pre- and post-intervention curves "
        "per record (matched colors, solid for pre / dashed for post).",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific EvalRecord indices to plot. Default: first --limit records that have data.",
    )
    p.add_argument("--limit", type=int, default=10, help="Max records to plot when --indices is omitted.")
    p.add_argument("--smoothing-window", type=int, default=1)
    p.add_argument("--output", required=True)
    return p.parse_args()


def short_label(prefix: str, idx: int, answer: str | None, max_chars: int = 24) -> str:
    if not answer:
        return f"{prefix} rec {idx}"
    snippet = answer.replace("\n", " ").strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return f"{prefix} rec {idx}: {snippet}"


def _has_field(record: dict, field: str) -> bool:
    if field == "both":
        return bool(record.get("completion_entropies")) and bool(record.get("intervened_completion_entropies"))
    return bool(record.get(f"{field}_entropies"))


def main() -> None:
    args = parse_args()
    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = report.get("records") or []
    with_data = [r for r in records if _has_field(r, args.field)]
    if not with_data:
        if args.field == "both":
            raise SystemExit(
                f"No records in {args.report} have both completion_entropies and "
                f"intervened_completion_entropies. Was the report produced with "
                f"--save-entropies AND --intervention-level > 0?"
            )
        raise SystemExit(
            f"No records in {args.report} have field '{args.field}_entropies'. "
            f"Re-run evaluate.py with --save-entropies."
        )

    if args.indices is not None:
        wanted = set(args.indices)
        chosen = [r for r in with_data if r.get("index") in wanted]
        missing = wanted - {r["index"] for r in chosen}
        if missing:
            print(f"warning: no entropy data for indices {sorted(missing)}")
    else:
        chosen = with_data[: args.limit]

    if not chosen:
        raise SystemExit("No matching records to plot.")

    series: list[tuple] = []
    for i, r in enumerate(chosen):
        color = f"C{i % 10}"
        markers = (r.get("completion_marker_positions") or None) if i == 0 else None
        if args.field == "both":
            series.append((
                short_label("pre", r["index"], r.get("predicted_answer")),
                r["completion_entropies"],
                {"color": color, "linestyle": "-"},
                markers,
            ))
            series.append((
                short_label("post", r["index"], r.get("intervened_answer")),
                r["intervened_completion_entropies"],
                {"color": color, "linestyle": "--"},
                None,
            ))
        elif args.field == "completion":
            series.append((
                short_label("completion", r["index"], r.get("predicted_answer")),
                r["completion_entropies"],
                {"color": color},
                markers,
            ))
        else:
            series.append((
                short_label("intervened_completion", r["index"], r.get("intervened_answer")),
                r["intervened_completion_entropies"],
                {"color": color},
                None,
            ))

    title = f'{report.get("model", "?")} · {args.field} · L{report.get("intervention_level", "?")}'
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_token_entropy_vs_index(
        series,
        args.output,
        smoothing_window=args.smoothing_window,
        title=title,
    )

    for item in series:
        label, entropies = item[0], item[1]
        mean = statistics.fmean(entropies)
        median = statistics.median(entropies)
        print(f"{label}  n={len(entropies)} mean={mean:.3f} median={median:.3f}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
