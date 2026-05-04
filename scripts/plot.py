import argparse
import json

from llm_faithfulness.plotting import (
    plot_faithfulness_vs_level,
    plot_metric_bars,
    plot_perf_vs_faithfulness,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot metrics from saved evaluation reports.")
    p.add_argument("--reports", nargs="+", required=True, help="One or more report JSON files")
    p.add_argument(
        "--kind",
        choices=["faithfulness_vs_level", "metric_bars", "perf_vs_faithfulness"],
        required=True,
    )
    p.add_argument("--metric", default="faithfulness_strong",
                   help="Metric for --kind metric_bars (faithfulness_id, faithfulness_strong, performance)")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.reports:
        with open(path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    if args.kind == "faithfulness_vs_level":
        plot_faithfulness_vs_level(reports, args.output)
    elif args.kind == "metric_bars":
        plot_metric_bars(reports, args.metric, args.output)
    else:
        plot_perf_vs_faithfulness(reports, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
