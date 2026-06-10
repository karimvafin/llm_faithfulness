import json
import os
import random
from typing import Iterable, Iterator

from .protocols import Dataset, InterventionStrategy
from .types import DPOPair, Entry

PairKind = str  # "full_response" | "sql_continuation" | "gold_vs_intervened" | "corrected_vs_unchanged"


def _reconstruct_answer(dataset: Dataset, mediator) -> str | None:
    if not hasattr(dataset, "reconstruct_answer"):
        return None
    return dataset.reconstruct_answer(mediator)


def _is_self_faithful(dataset: Dataset, entry: Entry, mediator, answer: str | None) -> bool:
    if not answer:
        return False
    db_schema = getattr(entry, "db_schema", None)
    if db_schema is None:
        return True
    try:
        from .pauq import sql_utils

        return sql_utils.faithfulness_id_check(mediator, answer, db_schema)
    except Exception:
        return False


def _answers_equivalent(entry: Entry, a: str | None, b: str | None) -> bool:
    if not a or not b:
        return a == b
    db_schema = getattr(entry, "db_schema", None)
    if db_schema is None:
        return a.strip() == b.strip()
    try:
        from .pauq import sql_utils

        return sql_utils.validate_sql(a, b, db_schema)
    except Exception:
        return a.strip() == b.strip()


def _make_full_response_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
    *,
    max_rejected_attempts: int,
) -> list[DPOPair]:
    """Match context_alignment's balanced DPO construction.

    For one valid edit, emit two pairs:
    - edit direction: edited mediator prefers edited answer over gold answer;
    - gold direction: gold mediator prefers gold answer over edited answer.
    """
    gold_med = dataset.gold_mediator(entry)
    gold_answer = dataset.gold_answer(entry)
    prompt = dataset.build_prompt(entry, gold_structure=False)

    for _ in range(max_rejected_attempts):
        intervened_med, _ = intervention.apply(entry, gold_med, level, rng)
        intervened_answer = _reconstruct_answer(dataset, intervened_med)
        if _answers_equivalent(entry, intervened_answer, gold_answer):
            continue
        if not _is_self_faithful(dataset, entry, intervened_med, intervened_answer):
            continue

        edit_chosen = dataset.render_mediator(intervened_med, answer=intervened_answer)
        edit_rejected = dataset.render_mediator(intervened_med, answer=gold_answer)
        gold_chosen = dataset.render_mediator(gold_med, answer=gold_answer)
        gold_rejected = dataset.render_mediator(gold_med, answer=intervened_answer)

        pairs: list[DPOPair] = []
        if edit_chosen != edit_rejected:
            pairs.append(DPOPair(
                prompt=prompt,
                chosen=edit_chosen,
                rejected=edit_rejected,
                kind="full_response_edit",
            ))
        if gold_chosen != gold_rejected:
            pairs.append(DPOPair(
                prompt=prompt,
                chosen=gold_chosen,
                rejected=gold_rejected,
                kind="full_response_gold",
            ))
        return pairs

    return []


def make_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
    *,
    kinds: Iterable[PairKind] = ("full_response",),
    chosen_intervention_prob: float = 0.0,
    max_rejected_attempts: int = 5,
) -> list[DPOPair]:
    """Build DPO pairs for one entry. Both kinds share the same intervened mediator.

    full_response:
        Balanced context_alignment-style DPO:
        edit pair: edited M + edited SQL > edited M + gold SQL
        gold pair: gold M + gold SQL > gold M + edited SQL

        Edited SQL is included only if it is self-faithful to the edited mediator.

    gold_vs_intervened:
        chosen   = gold M + gold SQL          (faithful)
        rejected = intervened M + gold SQL    (M↔SQL link broken)

    corrected_vs_unchanged:
        chosen   = intervened M + SQL reconstructed from intervened M  (faithful)
        rejected = intervened M + gold SQL                              (M↔SQL link broken)

    sql_continuation:
        prompt   = user prompt + intervened mediator prefix
        chosen   = SQL reconstructed from intervened M
        rejected = gold SQL

        This matches the intervention step in evaluation, where the model is asked to
        continue from a fixed intervened mediator and should generate only SQL.
    """
    gold_med = dataset.gold_mediator(entry)
    gold_answer = dataset.gold_answer(entry)
    intervened_med, _ = intervention.apply(entry, gold_med, level, rng)
    prompt = dataset.build_prompt(entry, gold_structure=False)

    rejected_str = dataset.render_mediator(intervened_med, answer=gold_answer)

    pairs: list[DPOPair] = []
    requested = set(kinds)

    if "full_response" in requested:
        pairs.extend(_make_full_response_pairs(
            dataset,
            intervention,
            entry,
            level,
            rng,
            max_rejected_attempts=max_rejected_attempts,
        ))

    if "sql_continuation" in requested and hasattr(dataset, "reconstruct_answer"):
        corrected_sql = _reconstruct_answer(dataset, intervened_med)
        assistant_prefix = dataset.render_mediator(intervened_med, answer=None) + "\n"
        if corrected_sql and corrected_sql != gold_answer:
            pairs.append(
                DPOPair(
                    prompt=prompt,
                    assistant_prefix=assistant_prefix,
                    chosen=corrected_sql,
                    rejected=gold_answer,
                    kind="sql_continuation",
                )
            )

    if "gold_vs_intervened" in requested:
        chosen_str = dataset.render_mediator(gold_med, answer=gold_answer)
        if chosen_str != rejected_str:
            pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=chosen_str,
                    rejected=rejected_str,
                    kind="gold_vs_intervened",
                )
            )

    if "corrected_vs_unchanged" in requested and hasattr(dataset, "reconstruct_answer"):
        corrected_sql = _reconstruct_answer(dataset, intervened_med)
        if corrected_sql and corrected_sql != gold_answer:
            chosen_str = dataset.render_mediator(intervened_med, answer=corrected_sql)
            if chosen_str != rejected_str:
                pairs.append(
                    DPOPair(
                        prompt=prompt,
                        chosen=chosen_str,
                        rejected=rejected_str,
                        kind="corrected_vs_unchanged",
                    )
                )

    return pairs


def iter_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    *,
    level: int,
    rng: random.Random,
    limit: int | None = None,
    sample_limit: int | None = None,
    kinds: Iterable[PairKind] = ("full_response",),
    chosen_intervention_prob: float = 0.0,
    max_rejected_attempts: int = 5,
) -> Iterator[DPOPair]:
    n = 0
    for sample_idx, entry in enumerate(dataset):
        if sample_limit is not None and sample_idx >= sample_limit:
            break
        if limit is not None and n >= limit:
            break
        for pair in make_dpo_pairs(
            dataset,
            intervention,
            entry,
            level,
            rng,
            kinds=kinds,
            chosen_intervention_prob=chosen_intervention_prob,
            max_rejected_attempts=max_rejected_attempts,
        ):
            yield pair
            n += 1
            if limit is not None and n >= limit:
                return


def write_jsonl(pairs: Iterator[DPOPair], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            row = {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            if p.assistant_prefix is not None:
                row["assistant_prefix"] = p.assistant_prefix
            if p.kind is not None:
                row["kind"] = p.kind
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
