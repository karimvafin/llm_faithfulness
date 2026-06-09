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


def _make_full_response_pair(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
    *,
    chosen_intervention_prob: float,
    max_rejected_attempts: int,
) -> DPOPair | None:
    """Match the old project's DPO construction.

    Chosen is a faithful full assistant response, either gold or intervened.
    Rejected uses an intervened mediator but keeps chosen's SQL, breaking the
    mediator/SQL correspondence.
    """
    gold_med = dataset.gold_mediator(entry)
    gold_answer = dataset.gold_answer(entry)
    prompt = dataset.build_prompt(entry, gold_structure=False)

    chosen_med = gold_med
    chosen_answer = gold_answer
    if rng.random() < chosen_intervention_prob:
        intervened_med, _ = intervention.apply(entry, gold_med, level, rng)
        intervened_answer = _reconstruct_answer(dataset, intervened_med)
        if intervened_answer:
            chosen_med = intervened_med
            chosen_answer = intervened_answer

    chosen_str = dataset.render_mediator(chosen_med, answer=chosen_answer)

    for _ in range(max_rejected_attempts):
        rejected_med, _ = intervention.apply(entry, gold_med, level, rng)
        rejected_answer = _reconstruct_answer(dataset, rejected_med)
        if not rejected_answer or rejected_answer == chosen_answer:
            continue
        rejected_str = dataset.render_mediator(rejected_med, answer=chosen_answer)
        if rejected_str != chosen_str:
            return DPOPair(
                prompt=prompt,
                chosen=chosen_str,
                rejected=rejected_str,
                kind="full_response",
            )

    return None


def make_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
    *,
    kinds: Iterable[PairKind] = ("full_response",),
    chosen_intervention_prob: float = 0.5,
    max_rejected_attempts: int = 5,
) -> list[DPOPair]:
    """Build DPO pairs for one entry. Both kinds share the same intervened mediator.

    full_response:
        chosen   = faithful full response (gold or intervened)
        rejected = intervened M + chosen SQL (M↔SQL link broken)

        This matches ../breaking-the-chain-intervention/prepare_dpo_data.py.

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
        pair = _make_full_response_pair(
            dataset,
            intervention,
            entry,
            level,
            rng,
            chosen_intervention_prob=chosen_intervention_prob,
            max_rejected_attempts=max_rejected_attempts,
        )
        if pair is not None:
            pairs.append(pair)

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
    kinds: Iterable[PairKind] = ("full_response",),
    chosen_intervention_prob: float = 0.5,
    max_rejected_attempts: int = 5,
) -> Iterator[DPOPair]:
    n = 0
    for entry in dataset:
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
