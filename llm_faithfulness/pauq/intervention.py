import random

from .types import PAUQEntry, PAUQIntervention, PAUQMediator, SlotKind


def _classify(value: str, db_schema: dict[str, list[str]]) -> SlotKind:
    v = value.lower()
    if v in db_schema:
        return "table"
    if any(v in cols for cols in db_schema.values()):
        return "column"
    return "literal"


def _all_columns(db_schema: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cols in db_schema.values():
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _pick_replacement(
    original: str,
    kind: SlotKind,
    db_schema: dict[str, list[str]],
    rng: random.Random,
) -> str | None:
    original_lc = original.lower()
    if kind == "table":
        candidates = [t for t in db_schema if t != original_lc]
    elif kind == "column":
        candidates = [c for c in _all_columns(db_schema) if c != original_lc]
    else:
        return None
    if not candidates:
        return None
    return rng.choice(candidates)


def _swap_table(links: dict[str, list[str]], before: str, after: str) -> None:
    """Rename table key before → after, merging columns if `after` already exists."""
    if before not in links:
        return
    cols = links.pop(before)
    if after in links:
        seen = set(links[after])
        for c in cols:
            if c not in seen:
                links[after].append(c)
                seen.add(c)
    else:
        links[after] = cols


def _swap_column(links: dict[str, list[str]], before: str, after: str) -> None:
    """Replace `before` with `after` in every table's column list (deduped)."""
    for t in links:
        if before not in links[t]:
            continue
        new_cols: list[str] = []
        seen: set[str] = set()
        for c in links[t]:
            new_c = after if c == before else c
            if new_c not in seen:
                new_cols.append(new_c)
                seen.add(new_c)
        links[t] = new_cols


class PAUQInterventionStrategy:
    def apply(
        self,
        entry: PAUQEntry,
        mediator: PAUQMediator,
        level: int,
        rng: random.Random,
    ) -> tuple[PAUQMediator, PAUQIntervention]:
        if not 0 <= level <= 5:
            raise ValueError(f"intervention level must be in [0, 5], got {level}")

        slots = list(mediator.slots)
        kinds: list[SlotKind] = [_classify(v, entry.db_schema) for v in slots]

        if level == 0 or not slots:
            return (
                PAUQMediator(skeleton=mediator.skeleton, schema_links=dict(mediator.schema_links), slots=slots),
                PAUQIntervention(level=level),
            )

        n_replace = (level * len(slots)) // 5
        candidate_idx = [i for i, k in enumerate(kinds) if k != "literal"]
        rng.shuffle(candidate_idx)
        chosen = sorted(candidate_idx[:n_replace])

        replaced_indices: list[int] = []
        before: list[str] = []
        after: list[str] = []
        kinds_recorded: list[SlotKind] = []

        new_slots = list(slots)
        new_links = {t: list(cols) for t, cols in mediator.schema_links.items()}

        for i in chosen:
            original = slots[i]
            kind = kinds[i]
            repl = _pick_replacement(original, kind, entry.db_schema, rng)
            if repl is None:
                continue

            for j in range(len(new_slots)):
                if slots[j] == original:
                    new_slots[j] = repl

            if kind == "table":
                _swap_table(new_links, original, repl)
            elif kind == "column":
                _swap_column(new_links, original, repl)

            replaced_indices.append(i)
            before.append(original)
            after.append(repl)
            kinds_recorded.append(kind)

        new_med = PAUQMediator(skeleton=mediator.skeleton, schema_links=new_links, slots=new_slots)
        rec = PAUQIntervention(
            level=level,
            replaced_slot_indices=replaced_indices,
            before=before,
            after=after,
            kinds=kinds_recorded,
        )
        return new_med, rec
