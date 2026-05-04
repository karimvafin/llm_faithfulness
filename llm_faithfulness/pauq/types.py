from dataclasses import dataclass, field
from typing import Literal

SlotKind = Literal["table", "column", "literal"]


@dataclass(slots=True)
class PAUQMediator:
    skeleton: str
    schema_links: dict[str, list[str]]
    slots: list[str]


@dataclass(slots=True)
class PAUQEntry:
    index: int
    question: str
    gold_sql: str
    db_id: str
    db_schema: dict[str, list[str]]
    gold_mediator: PAUQMediator


@dataclass(slots=True)
class PAUQIntervention:
    level: int
    replaced_slot_indices: list[int] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    kinds: list[SlotKind] = field(default_factory=list)
