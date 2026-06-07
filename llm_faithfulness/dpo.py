import json
import os
import random
from typing import Iterable, Iterator

from .protocols import Dataset, InterventionStrategy
from .types import DPOPair, Entry

PairKind = str  # "gold_vs_intervened" | "corrected_vs_unchanged"


def make_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
    *,
    kinds: Iterable[PairKind] = ("gold_vs_intervened", "corrected_vs_unchanged"),
) -> list[DPOPair]:
    """Build DPO pairs for one entry. Both kinds share the same intervened mediator.

    gold_vs_intervened:
        chosen   = gold M + gold SQL          (faithful)
        rejected = intervened M + gold SQL    (M↔SQL link broken)

    corrected_vs_unchanged:
        chosen   = intervened M + SQL reconstructed from intervened M  (faithful)
        rejected = intervened M + gold SQL                              (M↔SQL link broken)
    """
    gold_med = dataset.gold_mediator(entry)
    gold_answer = dataset.gold_answer(entry)
    intervened_med, _ = intervention.apply(entry, gold_med, level, rng)
    prompt = dataset.build_prompt(entry, gold_structure=False)

    rejected_str = dataset.render_mediator(intervened_med, answer=gold_answer)

    pairs: list[DPOPair] = []

    if "gold_vs_intervened" in kinds:
        chosen_str = dataset.render_mediator(gold_med, answer=gold_answer)
        if chosen_str != rejected_str:
            pairs.append(DPOPair(prompt=prompt, chosen=chosen_str, rejected=rejected_str))

    if "corrected_vs_unchanged" in kinds and hasattr(dataset, "reconstruct_answer"):
        corrected_sql = dataset.reconstruct_answer(intervened_med)
        if corrected_sql and corrected_sql != gold_answer:
            chosen_str = dataset.render_mediator(intervened_med, answer=corrected_sql)
            if chosen_str != rejected_str:
                pairs.append(DPOPair(prompt=prompt, chosen=chosen_str, rejected=rejected_str))

    return pairs


def iter_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    *,
    level: int,
    rng: random.Random,
    limit: int | None = None,
    kinds: Iterable[PairKind] = ("gold_vs_intervened", "corrected_vs_unchanged"),
) -> Iterator[DPOPair]:
    n = 0
    for entry in dataset:
        if limit is not None and n >= limit:
            break
        for pair in make_dpo_pairs(dataset, intervention, entry, level, rng, kinds=kinds):
            yield pair
            n += 1
            if limit is not None and n >= limit:
                return


def write_jsonl(pairs: Iterator[DPOPair], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}, ensure_ascii=False) + "\n")
            n += 1
    return n
