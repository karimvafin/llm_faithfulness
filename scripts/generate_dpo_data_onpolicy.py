import argparse
import random

from llm_faithfulness.dpo import write_jsonl
from llm_faithfulness.llm import LLM
from llm_faithfulness.pauq import sql_utils
from llm_faithfulness.pauq.dataset import PAUQDataset
from llm_faithfulness.pauq.intervention import PAUQInterventionStrategy
from llm_faithfulness.pauq.types import PAUQMediator
from llm_faithfulness.pipeline import run_evaluation
from llm_faithfulness.types import DPOPair

# On-policy DPO data.
#
# Off-policy pairs (gold/intervened mediator + gold SQL) never touch the policy's own
# mistakes, so DPO learns the preference without changing free generation. Here the
# REJECTED is the model's own unfaithful generation (its actual error distribution),
# and the CHOSEN is the same generation made self-consistent in one of two directions:
#
#   fix_mediator: keep the model's SQL, rebuild the mediator (skeleton/slots/schema_links)
#                 from that SQL. Targets the dominant failure (schema_links that don't
#                 match the SQL the model actually wrote).
#   fix_sql:      keep the model's mediator, regenerate the SQL from it (reconstruct).
#                 Targets "the SQL doesn't follow the declared mediator".
#
# A candidate is emitted only when CHOSEN actually passes faithfulness_id_check and
# differs from REJECTED, so every chosen target is a genuine faithful example.

_KINDS = ("fix_mediator", "fix_sql")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ON-POLICY DPO pairs from a model's own PAUQ generations.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-path", required=True, help="Folder with pauq_{train,dev}.json + tables.json")
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Limit number of dataset entries to generate from.")
    p.add_argument("--max-pairs", type=int, default=None, help="Stop after writing this many pairs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-api", action="store_true")
    p.add_argument("--api-base-url", default=None)
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=list(_KINDS),
        default=list(_KINDS),
        help="Which correction directions to emit per unfaithful generation.",
    )
    return p.parse_args()


def _mediator_from_sql(sql: str, db_schema: dict[str, list[str]]) -> PAUQMediator | None:
    """Rebuild a mediator that is consistent with `sql` (same construction the metric uses)."""
    try:
        skeleton, slots = sql_utils.extract_skeleton_and_slots(sql, db_schema)
    except Exception:
        return None
    parsed = sql_utils.parse_sql(sql, db_schema)
    links = sql_utils.extract_schema_links(parsed)
    if not links:
        links = sql_utils.schema_links_from_slots(slots, db_schema)
    return PAUQMediator(skeleton=skeleton, schema_links=links, slots=slots)


def _build_pairs_for_record(
    dataset: PAUQDataset,
    entry,
    med: PAUQMediator,
    sql: str,
    kinds: set[str],
) -> list[DPOPair]:
    db_schema = entry.db_schema
    prompt = dataset.build_prompt(entry, gold_structure=False)
    rejected = dataset.render_mediator(med, answer=sql)  # the model's own unfaithful block

    pairs: list[DPOPair] = []

    if "fix_mediator" in kinds:
        fixed_med = _mediator_from_sql(sql, db_schema)
        if fixed_med is not None and sql_utils.faithfulness_id_check(fixed_med, sql, db_schema):
            chosen = dataset.render_mediator(fixed_med, answer=sql)
            if chosen != rejected:
                pairs.append(DPOPair(prompt=prompt, chosen=chosen, rejected=rejected, kind="onpolicy_fix_mediator"))

    if "fix_sql" in kinds:
        fixed_sql = dataset.reconstruct_answer(med)
        if fixed_sql and sql_utils.faithfulness_id_check(med, fixed_sql, db_schema):
            chosen = dataset.render_mediator(med, answer=fixed_sql)
            if chosen != rejected:
                pairs.append(DPOPair(prompt=prompt, chosen=chosen, rejected=rejected, kind="onpolicy_fix_sql"))

    return pairs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset = PAUQDataset(args.data_path, split=args.split)
    intervention = PAUQInterventionStrategy()
    llm = LLM(args.model_name, use_api=args.use_api, api_base_url=args.api_base_url)

    # level=0 -> the model's own free generation, exactly as scored at eval (no intervention step).
    records = run_evaluation(
        dataset,
        intervention,
        llm,
        mode="structure_prediction",
        level=0,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        limit=args.limit,
    )
    entries_by_index = {e.index: e for e in dataset}

    kinds = set(args.kinds)
    n_unfaithful = 0
    n_faithful = 0
    n_unparsed = 0
    kind_counts = {"onpolicy_fix_mediator": 0, "onpolicy_fix_sql": 0}

    def _all_pairs():
        nonlocal n_unfaithful, n_faithful, n_unparsed
        for rec in records:
            med = rec.predicted_mediator
            sql = rec.predicted_answer
            if med is None or not sql:
                n_unparsed += 1
                continue
            entry = entries_by_index[rec.index]
            if sql_utils.faithfulness_id_check(med, sql, entry.db_schema):
                n_faithful += 1  # already self-consistent; nothing to correct
                continue
            n_unfaithful += 1
            pairs = _build_pairs_for_record(dataset, entry, med, sql, kinds)
            rng.shuffle(pairs)
            for pair in pairs:
                kind_counts[pair.kind] += 1
                yield pair

    written = 0

    def _capped():
        nonlocal written
        for pair in _all_pairs():
            if args.max_pairs is not None and written >= args.max_pairs:
                return
            written += 1
            yield pair

    n = write_jsonl(_capped(), args.output)

    print(f"generations: faithful={n_faithful} unfaithful={n_unfaithful} unparsed={n_unparsed}")
    print(f"pairs by kind: {kind_counts}")
    print(f"wrote {n} pairs -> {args.output}")


if __name__ == "__main__":
    main()
