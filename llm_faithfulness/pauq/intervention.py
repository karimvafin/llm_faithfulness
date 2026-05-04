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


def _recompute_schema_links(slots: list[str], db_schema: dict[str, list[str]]) -> dict[str, list[str]]:
    """Walk slots, classify, and (re)build schema_links so the mediator is self-consistent."""
    out: dict[str, list[str]] = {}
    column_to_tables: dict[str, list[str]] = {}
    for table, cols in db_schema.items():
        for col in cols:
            column_to_tables.setdefault(col, []).append(table)

    for value in slots:
        v_lc = value.lower()
        if v_lc in db_schema:
            out.setdefault(v_lc, [])
        elif v_lc in column_to_tables:
            for table in column_to_tables[v_lc]:
                cols = out.setdefault(table, [])
                if v_lc not in cols:
                    cols.append(v_lc)

    return {t: sorted(set(cs)) for t, cs in out.items()}


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
        for i in chosen:
            original = new_slots[i]
            kind = kinds[i]
            repl = _pick_replacement(original, kind, entry.db_schema, rng)
            if repl is None:
                continue
            for j, v in enumerate(new_slots):
                if v == original:
                    new_slots[j] = repl
            replaced_indices.append(i)
            before.append(original)
            after.append(repl)
            kinds_recorded.append(kind)

        new_links = _recompute_schema_links(new_slots, entry.db_schema)
        new_med = PAUQMediator(skeleton=mediator.skeleton, schema_links=new_links, slots=new_slots)
        rec = PAUQIntervention(
            level=level,
            replaced_slot_indices=replaced_indices,
            before=before,
            after=after,
            kinds=kinds_recorded,
        )
        return new_med, rec
