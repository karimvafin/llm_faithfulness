from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["structure_prediction", "gold_structure"]

Entry = Any
Mediator = Any
Intervention = Any


@dataclass(slots=True)
class EvalRecord:
    index: int
    mode: Mode
    intervention_level: int
    prompt: str
    completion: str
    predicted_mediator: Mediator | None
    predicted_answer: str | None
    intervention: Intervention | None = None
    intervened_mediator: Mediator | None = None
    intervened_prompt: str | None = None
    intervened_completion: str | None = None
    intervened_answer: str | None = None
    f_id: bool | None = None
    f_strong: bool | None = None
    perf: bool | None = None
    completion_entropies: list[float] | None = None
    intervened_completion_entropies: list[float] | None = None
    completion_marker_positions: dict[str, int] | None = None


@dataclass(slots=True)
class MetricsReport:
    model: str
    dataset: str
    mode: Mode
    intervention_level: int
    n: int
    faithfulness_id: float
    faithfulness_strong: float
    performance: float
    records: list[EvalRecord] = field(default_factory=list)


@dataclass(slots=True)
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
