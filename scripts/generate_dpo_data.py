import argparse
import random

from llm_faithfulness.dpo import iter_dpo_pairs, write_jsonl
from llm_faithfulness.pauq.dataset import PAUQDataset
from llm_faithfulness.pauq.intervention import PAUQInterventionStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DPO training pairs from PAUQ.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--intervention-level", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.intervention_level < 1:
        raise SystemExit("intervention level must be >= 1 for DPO data (level 0 ⇒ chosen == rejected)")

    dataset = PAUQDataset(args.data_path, split=args.split)
    intervention = PAUQInterventionStrategy()
    rng = random.Random(args.seed)

    pairs = iter_dpo_pairs(dataset, intervention, level=args.intervention_level, rng=rng, limit=args.limit)
    n = write_jsonl(pairs, args.output)
    print(f"wrote {n} pairs → {args.output}")


if __name__ == "__main__":
    main()
