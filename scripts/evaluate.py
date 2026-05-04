import argparse
import os

from llm_faithfulness.llm import LLM
from llm_faithfulness.metrics import aggregate, save_report, score_records
from llm_faithfulness.pauq.dataset import PAUQDataset
from llm_faithfulness.pauq.intervention import PAUQInterventionStrategy
from llm_faithfulness.pipeline import run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model on PAUQ.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-path", required=True, help="Folder with pauq_{train,dev}.json + tables.json")
    p.add_argument("--split", choices=["train", "dev"], default="dev")
    p.add_argument("--mode", choices=["structure_prediction", "gold_structure"], default="structure_prediction")
    p.add_argument("--intervention-level", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-api", action="store_true")
    p.add_argument("--api-base-url", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PAUQDataset(args.data_path, split=args.split)
    intervention = PAUQInterventionStrategy()
    llm = LLM(args.model_name, use_api=args.use_api, api_base_url=args.api_base_url)

    records = run_evaluation(
        dataset,
        intervention,
        llm,
        mode=args.mode,
        level=args.intervention_level,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        limit=args.limit,
    )
    entries_by_index = {e.index: e for e in dataset}
    score_records(dataset, entries_by_index, records)
    report = aggregate(
        records,
        model=args.model_name,
        dataset_name=dataset.name,
        mode=args.mode,
        intervention_level=args.intervention_level,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_report(report, args.output)
    print(
        f"n={report.n} "
        f"f_id={report.faithfulness_id:.3f} "
        f"f_strong={report.faithfulness_strong:.3f} "
        f"perf={report.performance:.3f}"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
