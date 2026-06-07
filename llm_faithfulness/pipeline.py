import random
import re
from typing import Iterable

from tqdm import tqdm

from .llm import LLM
from .protocols import Dataset, InterventionStrategy
from .types import EvalRecord, Entry, Mode

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks emitted by reasoning models (Qwen3, etc.)."""
    return _THINK_BLOCK.sub("", text).lstrip()


def _build_continuation_prompt(llm: LLM, dataset: Dataset, entry: Entry, mediator) -> str:
    """Build a prompt where the user message is the standard prompt and the assistant
    prefix is the rendered mediator (without the SQL line). The model then continues
    by producing only the SQL."""
    user = dataset.build_prompt(entry, gold_structure=False)
    base = llm.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=False,
    )
    return base + dataset.render_mediator(mediator, answer=None) + "\n"


def _build_fresh_prompt(llm: LLM, dataset: Dataset, entry: Entry) -> str:
    user = dataset.build_prompt(entry, gold_structure=False)
    return llm.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
    )


def _truncate_sql(text: str) -> str | None:
    s = text.strip()
    if not s:
        return None
    if ";" in s:
        s = s[:s.index(";") + 1]
    return s or None


def _batched(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_evaluation(
    dataset: Dataset,
    intervention: InterventionStrategy,
    llm: LLM,
    *,
    mode: Mode,
    level: int,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    seed: int = 42,
    limit: int | None = None,
    save_entropies: bool = False,
) -> list[EvalRecord]:
    rng = random.Random(seed)
    entries: list[Entry] = list(dataset)
    if limit is not None:
        entries = entries[:limit]

    records: list[EvalRecord] = []
    for batch in tqdm(list(_batched(entries, batch_size)), desc="generate"):
        records.extend(
            _run_batch(
                dataset, intervention, llm, batch, mode, level, rng,
                max_new_tokens, save_entropies,
            )
        )
    return records


def _generate(
    llm: LLM,
    prompts: list[str],
    max_new_tokens: int,
    save_entropies: bool,
    markers: list[str] | None = None,
) -> tuple[list[str], list[list[float] | None], list[dict[str, int] | None]]:
    """Returns (raw_completions, entropies_per_prompt, marker_positions_per_prompt).
    entropies / marker_positions are None when not requested."""
    if save_entropies:
        results = llm.generate_with_entropy(prompts, max_new_tokens=max_new_tokens, markers=markers)
        texts = [t for t, _, _ in results]
        ents: list[list[float] | None] = [e for _, e, _ in results]
        mps: list[dict[str, int] | None] = [mp for _, _, mp in results]
        return texts, ents, mps
    texts = llm.generate(prompts, max_new_tokens=max_new_tokens)
    return texts, [None] * len(texts), [None] * len(texts)


def _run_batch(
    dataset: Dataset,
    intervention: InterventionStrategy,
    llm: LLM,
    batch: list[Entry],
    mode: Mode,
    level: int,
    rng: random.Random,
    max_new_tokens: int,
    save_entropies: bool,
) -> list[EvalRecord]:
    dataset_markers = getattr(dataset, "section_markers", None)
    if mode == "gold_structure":
        prompts = [_build_continuation_prompt(llm, dataset, e, dataset.gold_mediator(e)) for e in batch]
        raw_completions, entropies, marker_positions = _generate(
            llm, prompts, max_new_tokens, save_entropies, markers=dataset_markers,
        )
        completions = [_strip_thinking(c) for c in raw_completions]
        records = [
            EvalRecord(
                index=e.index,
                mode=mode,
                intervention_level=level,
                prompt=p,
                completion=c,
                predicted_mediator=dataset.gold_mediator(e),
                predicted_answer=_truncate_sql(c),
                completion_entropies=ent,
                completion_marker_positions=mp,
            )
            for e, p, c, ent, mp in zip(batch, prompts, completions, entropies, marker_positions)
        ]
    else:
        prompts = [_build_fresh_prompt(llm, dataset, e) for e in batch]
        raw_completions, entropies, marker_positions = _generate(
            llm, prompts, max_new_tokens, save_entropies, markers=dataset_markers,
        )
        completions = [_strip_thinking(c) for c in raw_completions]
        records = []
        for e, p, c, ent, mp in zip(batch, prompts, completions, entropies, marker_positions):
            med, sql = dataset.parse_completion(c)
            records.append(
                EvalRecord(
                    index=e.index,
                    mode=mode,
                    intervention_level=level,
                    prompt=p,
                    completion=c,
                    predicted_mediator=med,
                    predicted_answer=sql,
                    completion_entropies=ent,
                    completion_marker_positions=mp,
                )
            )

    if level > 0:
        _apply_intervention_step(
            dataset, intervention, llm, batch, records, level, rng, max_new_tokens, save_entropies,
        )

    return records


def _apply_intervention_step(
    dataset: Dataset,
    intervention: InterventionStrategy,
    llm: LLM,
    batch: list[Entry],
    records: list[EvalRecord],
    level: int,
    rng: random.Random,
    max_new_tokens: int,
    save_entropies: bool,
) -> None:
    intervened_prompts: list[str] = []
    valid_idx: list[int] = []
    for i, (entry, rec) in enumerate(zip(batch, records)):
        base_med = rec.predicted_mediator
        if base_med is None:
            continue
        new_med, inter = intervention.apply(entry, base_med, level, rng)
        rec.intervention = inter
        rec.intervened_mediator = new_med
        prompt = _build_continuation_prompt(llm, dataset, entry, new_med)
        rec.intervened_prompt = prompt
        intervened_prompts.append(prompt)
        valid_idx.append(i)

    if not intervened_prompts:
        return

    raw_completions, entropies, _ = _generate(
        llm, intervened_prompts, max_new_tokens, save_entropies, markers=None,
    )
    completions = [_strip_thinking(c) for c in raw_completions]
    for i, completion, ent in zip(valid_idx, completions, entropies):
        rec = records[i]
        rec.intervened_completion = completion
        rec.intervened_answer = _truncate_sql(completion)
        rec.intervened_completion_entropies = ent
