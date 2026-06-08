import argparse
import json
import os

from llm_faithfulness.plotting import plot_performance_fid_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grouped bar plot: Performance vs F_ID for model comparison."
    )
    p.add_argument("--reports", nargs="+", required=True, help="One or more evaluation report JSON files")
    p.add_argument("--output", required=True, help="Path to output PNG/PDF")
    p.add_argument(
        "--mode",
        default="structure_prediction",
        help="Filter by report mode. Use 'all' to keep all modes.",
    )
    p.add_argument(
        "--intervention-level",
        type=int,
        default=0,
        help="Use reports at this intervention level (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports: list[dict] = []
    for path in args.reports:
        with open(path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    mode = None if args.mode == "all" else args.mode
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_performance_fid_bars(
        reports,
        args.output,
        mode=mode,
        intervention_level=args.intervention_level,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
