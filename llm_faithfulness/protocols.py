import random
from typing import Iterator, Protocol, runtime_checkable

from .types import Entry, EvalRecord, Intervention, Mediator


@runtime_checkable
class Dataset(Protocol):
    name: str

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Entry]: ...

    def build_prompt(self, entry: Entry, *, gold_structure: bool) -> str: ...
    def build_continuation_prompt(self, entry: Entry, mediator: Mediator) -> str: ...

    def parse_completion(self, text: str) -> tuple[Mediator | None, str | None]: ...

    def render_mediator(self, mediator: Mediator, answer: str | None = None) -> str: ...

    def score(self, entry: Entry, record: EvalRecord) -> dict[str, bool | None]: ...

    def gold_mediator(self, entry: Entry) -> Mediator: ...
    def gold_answer(self, entry: Entry) -> str: ...


@runtime_checkable
class InterventionStrategy(Protocol):
    def apply(
        self,
        entry: Entry,
        mediator: Mediator,
        level: int,
        rng: random.Random,
    ) -> tuple[Mediator, Intervention]: ...
