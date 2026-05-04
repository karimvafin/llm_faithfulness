import json
import os
import random
from typing import Iterator

from .protocols import Dataset, InterventionStrategy
from .types import DPOPair, Entry


def make_dpo_pair(
    dataset: Dataset,
    intervention: InterventionStrategy,
    entry: Entry,
    level: int,
    rng: random.Random,
) -> DPOPair | None:
    """chosen = gold mediator + gold answer.
    rejected = level-k-intervened mediator + same gold answer (broken mediator↔answer link).
    Returns None if intervention produced an unchanged mediator."""
    gold_med = dataset.gold_mediator(entry)
    gold_answer = dataset.gold_answer(entry)
    intervened_med, _ = intervention.apply(entry, gold_med, level, rng)

    chosen = dataset.render_mediator(gold_med, answer=gold_answer)
    rejected = dataset.render_mediator(intervened_med, answer=gold_answer)
    if chosen == rejected:
        return None

    prompt = dataset.build_prompt(entry, gold_structure=False)
    return DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)


def iter_dpo_pairs(
    dataset: Dataset,
    intervention: InterventionStrategy,
    *,
    level: int,
    rng: random.Random,
    limit: int | None = None,
) -> Iterator[DPOPair]:
    n = 0
    for entry in dataset:
        if limit is not None and n >= limit:
            break
        pair = make_dpo_pair(dataset, intervention, entry, level, rng)
        if pair is None:
            continue
        yield pair
        n += 1


def write_jsonl(pairs: Iterator[DPOPair], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}, ensure_ascii=False) + "\n")
            n += 1
    return n
