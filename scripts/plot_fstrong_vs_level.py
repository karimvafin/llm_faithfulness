import argparse
import json
import os

from llm_faithfulness.plotting import plot_fstrong_vs_intervention_level


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure 1: FStrong vs intervention level (one line per model)."
    )
    p.add_argument("--reports", nargs="+", required=True, help="One or more evaluation report JSON files")
    p.add_argument("--output", required=True, help="Path to output PNG/PDF")
    p.add_argument(
        "--mode",
        default="structure_prediction",
        help="Filter by report mode. Use 'all' to keep all modes.",
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
    plot_fstrong_vs_intervention_level(reports, args.output, mode=mode)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
