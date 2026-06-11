import argparse
import json
import os

from llm_faithfulness.plotting import _pct_change, plot_dpo_before_after


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare F_ID and Performance before vs after DPO. Each --pair takes "
        "two report JSONs (before after); the plot annotates the relative change "
        "(how many % F_ID rose and how many % Performance fell)."
    )
    p.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BEFORE", "AFTER"),
        required=True,
        help="Two evaluation report JSONs: before-DPO then after-DPO. Repeatable.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional x-axis label per pair (defaults to the before report's model name).",
    )
    p.add_argument("--output", required=True, help="Path to output PNG/PDF")
    return p.parse_args()


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    pairs = [(_load(before), _load(after)) for before, after in args.pair]
    if args.labels is not None and len(args.labels) != len(pairs):
        raise SystemExit(f"--labels has {len(args.labels)} entries but there are {len(pairs)} pairs.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_dpo_before_after(pairs, args.output, labels=args.labels)

    # also print the numbers so they are available without opening the figure
    print(f"{'label':<28} {'F_ID before':>12} {'F_ID after':>11} {'ΔF_ID':>9}   "
          f"{'Perf before':>12} {'Perf after':>11} {'ΔPerf':>9}")
    for i, (b, a) in enumerate(pairs):
        label = (args.labels[i] if args.labels else str(b.get("model", "?")).split("/")[-1])[:28]
        fb, fa = float(b.get("faithfulness_id", 0.0)), float(a.get("faithfulness_id", 0.0))
        pb, pa = float(b.get("performance", 0.0)), float(a.get("performance", 0.0))

        def fmt(before: float, after: float) -> str:
            pct = _pct_change(before, after)
            return f"{after - before:+.3f}" if pct is None else f"{pct:+.1f}%"

        print(f"{label:<28} {fb:>12.3f} {fa:>11.3f} {fmt(fb, fa):>9}   "
              f"{pb:>12.3f} {pa:>11.3f} {fmt(pb, pa):>9}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
