import argparse
import random

from llm_faithfulness.dpo import iter_dpo_pairs, write_jsonl
from llm_faithfulness.pauq.dataset import PAUQDataset
from llm_faithfulness.pauq.intervention import PAUQInterventionStrategy


_ALL_KINDS = ("full_response", "sql_continuation", "gold_vs_intervened", "corrected_vs_unchanged")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DPO training pairs from PAUQ.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--intervention-level", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None, help="Maximum number of DPO pairs to write.")
    p.add_argument("--sample-limit", type=int, default=None, help="Maximum number of dataset samples to process.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pair-kinds",
        nargs="+",
        choices=list(_ALL_KINDS),
        default=["full_response"],
        help="Which DPO pair kinds to emit per entry. "
        "'full_response': emits two full-block pairs per valid intervention: "
        "X-M-Y > X-M'-Y and X-M'-Y' > X-M'-Y. "
        "'sql_continuation': prompt includes the intervened mediator prefix, "
        "chosen/rejected are SQL-only continuations (matches evaluation intervention). "
        "'gold_vs_intervened': chosen=gold M+SQL, rejected=intervened M+gold SQL. "
        "'corrected_vs_unchanged': chosen=intervened M+SQL reconstructed from M, "
        "rejected=intervened M+gold SQL.",
    )
    p.add_argument(
        "--chosen-intervention-prob",
        type=float,
        default=0.0,
        help="Deprecated; kept for compatibility. full_response now always uses balanced edit/gold pairs.",
    )
    p.add_argument(
        "--max-rejected-attempts",
        type=int,
        default=5,
        help="For full_response pairs, retries to find a mediator/SQL mismatch for rejected.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.intervention_level < 1:
        raise SystemExit("intervention level must be >= 1 for DPO data (level 0 ⇒ chosen == rejected)")

    dataset = PAUQDataset(args.data_path, split=args.split)
    intervention = PAUQInterventionStrategy()
    rng = random.Random(args.seed)

    pairs = iter_dpo_pairs(
        dataset, intervention,
        level=args.intervention_level, rng=rng, limit=args.limit, sample_limit=args.sample_limit,
        kinds=tuple(args.pair_kinds),
        chosen_intervention_prob=args.chosen_intervention_prob,
        max_rejected_attempts=args.max_rejected_attempts,
    )
    n = write_jsonl(pairs, args.output)
    print(f"wrote {n} pairs → {args.output} (kinds={args.pair_kinds})")


if __name__ == "__main__":
    main()
